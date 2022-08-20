import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, out_channels, kernel_size):

        super(Encoder, self).__init__()

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=kernel_size//2,
                                padding=0)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.Conv1d(x)  # torch.Size([1, 1, 32000]) => torch.Size([1, 64, 31999])

        x = self.ReLU(x)

        return x


class Segmentation(nn.Module):

    def __init__(self, K):

        super(Segmentation, self).__init__()

        self.K = K

    def forward(self, x):
        """
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """

        B, D, I = x.shape  # torch.Size([1, 128, 31999])

        P = self.K // 2  # 100 = 200 // 2

        input, gap = self._padding(x)  # 填充, torch.Size([1, 128, 32300]), 101

        del x

        input1 = input[:, :, :-P].contiguous().view(B, D, -1, self.K)  # torch.Size([1, 128, 161, 200])
        input2 = input[:, :, P:].contiguous().view(B, D, -1, self.K)  # torch.Size([1, 128, 161, 200])

        input = torch.cat([input1, input2], dim=3).view(B, D, -1, self.K).transpose(2, 3).contiguous()  # torch.Size([1, 128, 200, 322])

        return input.contiguous(), gap

    def _padding(self, x):
        """
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = x.shape  # torch.Size([1, 128, 31999])

        P = self.K // 2

        gap = self.K - (P + L % self.K) % self.K  # 200 - (100 + 31999 % 200) % 200 = 101

        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(x.type())  # torch.Size([1, 128, 101])
            x = torch.cat([x, pad], dim=2)  # torch.Size([1, 128, 32100])

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(x.type())  # torch.Size([1, 128, 100])

        x = torch.cat([_pad, x, _pad], dim=2)  # torch.Size([1, 128, 32300])

        return x, gap


class Locally_Recurrent(nn.Module):

    def __init__(self, in_channels, hidden_channels=128, num_layers=1, bidirectional=True):

        super(Locally_Recurrent, self).__init__()

        self.LSTM = nn.LSTM(input_size=in_channels,  # 输入的维度
                            hidden_size=hidden_channels,  # 隐藏层的维度
                            num_layers=num_layers,  # LSTM 的层数
                            bias=True,  # 偏置
                            batch_first=True,  # True, (batch, seq, feature); False, (seq, batch, feature)
                            bidirectional=bidirectional)  # 是否为双向 LSTM

        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)

        # 线性层
        self.Linear = nn.Linear(hidden_channels*2 if bidirectional else hidden_channels, in_channels)

    def forward(self, x):

        B, N, K, S = x.shape  # torch.Size([1, 128, 200, 322])

        x = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)  # 调整维度通过 LSTM, torch.Size([322, 200, 128])

        residual1 = x  # 残余连接

        x, _ = self.LSTM(x)  # 双向 LSTM, torch.Size([200, 322, 256])

        x = self.Linear(x)  # 线性层减少输出通道, torch.Size([200, 322, 128])

        x = self.LayerNorm(x + residual1)  # 层归一化

        del residual1

        x = x.view(B, S, K, N).permute(0, 3, 2, 1).contiguous()  # 恢复原来的维度, torch.Size([1, 128, 200, 322])

        return x


class Positional_Encoding(nn.Module):
    """
        Implement the positional encoding (PE) function.
        PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=32000):

        super(Positional_Encoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
            Args:
                input: N x T x D
        """
        length = input.size(1)

        return self.pe[:, :length]


class Globally_Attentive(nn.Module):

    def __init__(self, in_channels, num_heads=8):

        super(Globally_Attentive, self).__init__()

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=in_channels)

        self.Positional_Encoding = Positional_Encoding(d_model=in_channels,
                                                       max_len=32000)

        self.MultiheadAttention = nn.MultiheadAttention(embed_dim=in_channels,
                                                        num_heads=num_heads,
                                                        dropout=0.1,
                                                        batch_first=True)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=in_channels)

        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        B, N, K, S = x.shape  # torch.Size([1, 128, 200, 322])

        residual1 = x  # 残余连接, torch.Size([1, 128, 200, 322])

        x = x.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)  # 调整维度, 注意与循环网络区分, torch.Size([200, 322, 128])

        x = self.LayerNorm1(x) + self.Positional_Encoding(x)  # 加入位置信息, torch.Size([200, 322, 128])

        residual2 = x

        x = self.MultiheadAttention(x, x, x, attn_mask=None, key_padding_mask=None)[0]  # torch.Size([200, 322, 128])

        x = residual2 + self.Dropout(x)  # torch.Size([200, 322, 128])

        del residual2

        x = self.LayerNorm2(x)

        x = x.view(B, K, S, N).permute(0, 3, 1, 2).contiguous()  # torch.Size([1, 128, 200, 322])

        x = x + residual1  # torch.Size([1, 128, 200, 322])

        del residual1

        return x


class Sandglasset_Block(nn.Module):

    def __init__(self, in_channels, hidden_channels=128, num_layers=1, bidirectional=True, num_heads=8,
                 kernel_size=1, stride=1):

        super(Sandglasset_Block, self).__init__()

        self.Locally_Recurrent = Locally_Recurrent(in_channels=in_channels,
                                                   hidden_channels=hidden_channels,
                                                   num_layers=num_layers,
                                                   bidirectional=bidirectional)

        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)

        self.Conv1D = nn.Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0)

        self.Globally_Attentive = Globally_Attentive(in_channels=in_channels,
                                                     num_heads=num_heads)

        self.ConvTrans1D = nn.ConvTranspose1d(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=0)

    def forward(self, x):

        B, N, K, S = x.shape  # torch.Size([1, 64, 200, 322])

        residual = x

        x = self.Locally_Recurrent(x)  # torch.Size([1, 64, 200, 322])

        x = self.LayerNorm(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous() + residual

        # 下采样
        x = x.permute(0, 3, 1, 2).contiguous().view(B*S, N, K)
        x = self.Conv1D(x)
        x = x.view(B, S, N, -1).permute(0, 2, 3, 1).contiguous()

        x = self.Globally_Attentive(x)

        # 上采样
        x = x.permute(0, 3, 1, 2).contiguous().view(B*S, N, -1)
        x = self.ConvTrans1D(x)
        x = x.view(B, S, N, -1).permute(0, 2, 3, 1).contiguous()

        return x


class Separation(nn.Module):

    def __init__(self, in_channels, out_channels, length, hidden_channels=128, num_layers=1, bidirectional=True,
                 num_heads=8, cycle_amount=6, speakers=2):

        super(Separation, self).__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)

        self.Linear = nn.Linear(in_features=in_channels, out_features=out_channels)

        self.Segmentation = Segmentation(K=length)

        self.cycle_amount = cycle_amount

        kernel_size = []
        stride = []

        for i in range(self.cycle_amount):
            if i < self.cycle_amount//2:
                kernel_size.append(4**i)
                stride.append(4**i)
            else:
                kernel_size.append(4**(self.cycle_amount-i-1))
                stride.append(4**(self.cycle_amount-i-1))

        self.Sandglasset_Blocks = nn.ModuleList([])

        for i in range(self.cycle_amount):
            self.Sandglasset_Blocks.append(Sandglasset_Block(in_channels=out_channels,
                                                             hidden_channels=hidden_channels,
                                                             num_layers=num_layers,
                                                             bidirectional=bidirectional,
                                                             num_heads=num_heads,
                                                             kernel_size=kernel_size[i],
                                                             stride=stride[i]))

        self.residual = []

        self.PReLU = nn.PReLU()

        self.Spk = speakers

        self.Conv2d = nn.Conv2d(in_channels=out_channels,  # 64
                                out_channels=speakers*out_channels,  # 128
                                kernel_size=1)

        self.output = nn.Sequential(nn.Conv1d(in_channels=out_channels,  # 64
                                              out_channels=out_channels,   # 64
                                              kernel_size=1),  # 64
                                    nn.Tanh())

        self.output_gate = nn.Sequential(nn.Conv1d(in_channels=out_channels,  # 64
                                                   out_channels=out_channels,  # 64
                                                   kernel_size=1),
                                         nn.Sigmoid())

        self.Conv1x1 = nn.Conv1d(in_channels=out_channels,  # 64
                                 out_channels=in_channels,  # 128
                                 kernel_size=1,
                                 bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.LayerNorm(x.permute(0, 2, 1).contiguous())

        x = self.Linear(x).permute(0, 2, 1).contiguous()

        x, gap = self.Segmentation(x)  # torch.Size([3, 64, 256, 64])

        self.residual = []

        for i in range(self.cycle_amount):

            x = self.Sandglasset_Blocks[i](x)  # torch.Size([1, 64, 200, 322])

            if i < self.cycle_amount//2:
                self.residual.append(x)
            else:
                x = x + self.residual[self.cycle_amount-i-1]

        x = self.PReLU(x)

        x = self.Conv2d(x)  # 通道说话人个数倍, torch.Size([2, 128, 200, 322])

        B, _, K, S = x.shape  # torch.Size([1, 128, 200, 322])

        x = x.view(B * self.Spk, -1, K, S)  # torch.Size([2, 64, 200, 322])

        x = self._over_add(x, gap)  # torch.Size([2, 64, 31999])

        x = self.output(x) * self.output_gate(x)  # torch.Size([2, 64, 31999])

        x = self.Conv1x1(x)  # torch.Size([2, 128, 31999])

        _, N, L = x.shape

        x = x.view(B, self.Spk, N, L)  # torch.Size([1, 2, 128, 31999])

        x = self.ReLU(x)

        x = x.transpose(0, 1).contiguous()  # torch.Size([2, 1, 128, 31999])

        return x

    def _over_add(self, input, gap):
        """
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2

        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2

        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Decoder(nn.Module):

    def __init__(self, in_channels, kernel_size):

        super(Decoder, self).__init__()

        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=in_channels,  # 256
                                                  out_channels=1,
                                                  kernel_size=kernel_size,  # 16
                                                  stride=kernel_size//2,  # 8
                                                  padding=0)

    def forward(self, x):

        x = self.ConvTranspose1d(x)

        return x


class Sandglasset(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, length, hidden_channels=128, num_layers=1,
                 bidirectional=True, num_heads=8, cycle_amount=6, speakers=2):

        super(Sandglasset, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.kernel_size = kernel_size

        self.length = length

        self.hidden_channels = hidden_channels

        self.num_layers = num_layers

        self.bidirectional = bidirectional

        self.num_heads = num_heads

        self.cycle_amount = cycle_amount

        self.speakers = speakers

        self.Encoder = Encoder(out_channels=self.in_channels,
                               kernel_size=self.kernel_size)

        self.Separation = Separation(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     length=self.length,
                                     hidden_channels=self.hidden_channels,
                                     num_layers=self.num_layers,
                                     bidirectional=self.bidirectional,
                                     num_heads=self.num_heads,
                                     cycle_amount=self.cycle_amount,
                                     speakers=self.speakers)

        self.Spk = self.speakers

        self.Decoder = Decoder(in_channels=self.in_channels,
                               kernel_size=self.kernel_size)

    def forward(self, x):

        x, rest = self.pad_signal(x)

        e = self.Encoder(x)  # torch.Size([1, 32000]) => torch.Size([1, 128, 31999])

        s = self.Separation(e)  # torch.Size([2, 1, 128, 31999])

        out = [s[i]*e for i in range(self.Spk)]

        del e, s

        audio = [self.Decoder(out[i]) for i in range(self.Spk)]

        del out

        audio[0] = audio[0][:, :, self.kernel_size // 2:-(rest + self.kernel_size // 2)].contiguous()  # B, 1, T
        audio[1] = audio[1][:, :, self.kernel_size // 2:-(rest + self.kernel_size // 2)].contiguous()  # B, 1, T
        audio = torch.cat(audio, dim=1)  # [B, C, T]

        return audio

    def pad_signal(self, x):

        # 输入波形: (B, T) or (B, 1, T)
        # 调整和填充

        if x.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)  # 每一个批次的大小
        nsample = x.size(2)  # 单个数据的长度
        rest = self.kernel_size - (self.kernel_size // 2 + nsample % self.kernel_size) % self.kernel_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(x.type())
            x = torch.cat([x, pad], dim=2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.kernel_size // 2)).type(x.type())

        x = torch.cat([pad_aux, x, pad_aux], 2)

        return x, rest

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(in_channels=package['in_channels'], out_channels=package['out_channels'],
                    kernel_size=package['kernel_size'], length=package['length'],
                    hidden_channels=package['hidden_channels'], num_layers=package['num_layers'],
                    bidirectional=package['bidirectional'], num_heads=package['num_heads'],
                    cycle_amount=package['cycle_amount'], speakers=package['speakers'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'in_channels': model.in_channels, 'out_channels': model.out_channels,
            'kernel_size': model.kernel_size, 'length': model.length,
            'hidden_channels': model.hidden_channels, 'num_layers': model.num_layers,
            'bidirectional': model.bidirectional, 'num_heads': model.num_heads,
            'cycle_amount': model.cycle_amount, 'speakers': model.speakers,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == "__main__":

    x = torch.rand(1, 32000)

    model = Sandglasset(in_channels=256,
                        out_channels=64,
                        kernel_size=4,
                        length=256,
                        hidden_channels=128,
                        num_layers=1,
                        bidirectional=True,
                        num_heads=8,
                        cycle_amount=6,
                        speakers=2)

    y = model(x)

    print("{:.3f} million".format(sum([param.nelement() for param in model.parameters()]) / 1e6))

    print(y.shape)
