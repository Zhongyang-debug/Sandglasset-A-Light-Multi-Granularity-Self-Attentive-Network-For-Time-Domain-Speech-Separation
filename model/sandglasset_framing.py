import torch
from torch import nn
import math


class framing(nn.Module):

    def __init__(self, length):

        super(framing, self).__init__()

        self.length = length

    def forward(self, x):
        """
           M: window length
           P: hop size
           input: [B, L]
           output: [B, K, S]
        """

        B, L = x.shape

        P = self.length // 2

        input, gap = self._padding(x)

        del x

        input1 = input[:, :-P].reshape(B, -1, self.length)
        input2 = input[:, P:].reshape(B, -1, self.length)

        input = torch.cat([input1, input2], dim=2).reshape(B, -1, self.length).transpose(1, 2)

        return input, gap

    def _padding(self, x):
        """
           M: window length
           P: hop size
           input: [B, L]
        """

        B, L = x.shape  # torch.Size([1, 32000])

        P = self.length // 2  # hop size

        gap = self.length - (P + L % self.length) % self.length

        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, gap)).type(x.type())  # torch.Size([1, 2])
            x = torch.cat([x, pad], dim=1)  # torch.Size([1, 32002])

        _pad = torch.Tensor(torch.zeros(B, P)).type(x.type())  # torch.Size([1, 2])

        x = torch.cat([_pad, x, _pad], dim=1)  # torch.Size([1, 32006])

        return x, gap


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(Encoder, self).__init__()

        if kernel_size == 1:
            self.Conv1d = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        else:
            self.Conv1d = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=kernel_size//2,
                                    padding=0)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.Conv1d(x)

        x = self.ReLU(x)

        return x


class Segmentation(nn.Module):

    def __init__(self, length):

        super(Segmentation, self).__init__()

        self.length = length

    def forward(self, x):
        """
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """

        B, C, L = x.shape

        P = self.length // 2

        input, gap = self._padding(x)

        del x

        input1 = input[:, :, :-P].reshape(B, C, -1, self.length)
        input2 = input[:, :, P:].reshape(B, C, -1, self.length)

        input = torch.cat([input1, input2], dim=3).reshape(B, C, -1, self.length).transpose(2, 3)

        return input, gap

    def _padding(self, x):
        """
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, C, L = x.shape

        P = self.length // 2

        gap = self.length - (P + L % self.length) % self.length

        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, C, gap)).type(x.type())
            x = torch.cat([x, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, C, P)).type(x.type())

        x = torch.cat([_pad, x, _pad], dim=2)

        return x, gap


class Locally_Recurrent(nn.Module):

    def __init__(self, in_channels, hidden_channels=128, num_layers=1, bidirectional=True):

        super(Locally_Recurrent, self).__init__()

        self.Bi_LSTM = nn.LSTM(input_size=in_channels,  # 输入的维度
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

        x = x.permute(0, 3, 2, 1).reshape(B*S, K, N)  # 调整维度通过 LSTM, torch.Size([322, 200, 128])

        residual1 = x  # 残余连接

        x, _ = self.Bi_LSTM(x)  # 双向 LSTM, torch.Size([200, 322, 256])

        x = self.Linear(x)  # 线性层减少输出通道, torch.Size([200, 322, 128])

        x = self.LayerNorm(x + residual1)  # 层归一化

        del residual1

        x = x.reshape(B, S, K, N).permute(0, 3, 2, 1)  # 恢复原来的维度, torch.Size([1, 128, 200, 322])

        return x


class Positional_Encoding(nn.Module):
    """
        Implement the positional encoding (PE) function.
        PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):

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


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


class Globally_Attentive(nn.Module):

    def __init__(self, in_channels, num_heads=8):

        super(Globally_Attentive, self).__init__()

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=in_channels)

        self.Positional_Encoding = Positional_Encoding(d_model=in_channels,
                                                       max_len=8000)

        self.Self_Attention = Self_Attention(input_dim=in_channels, dim_k=in_channels, dim_v=in_channels)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=in_channels)

        self.Dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        B, N, K, S = x.shape  # torch.Size([1, 128, 200, 322])

        residual1 = x  # 残余连接, torch.Size([1, 128, 200, 322])

        x = x.permute(0, 2, 3, 1).reshape(B*K, S, N)  # 调整维度, 注意与循环网络区分，分别为局部和全局, torch.Size([200, 322, 128])

        x = self.LayerNorm1(x) + self.Positional_Encoding(x)  # 加入位置信息, torch.Size([200, 322, 128])

        residual2 = x

        x = self.Self_Attention(x)  # torch.Size([200, 322, 128])

        x = residual2 + self.Dropout(x)  # torch.Size([200, 322, 128])

        del residual2

        x = self.LayerNorm2(x)

        x = x.reshape(B, K, S, N).permute(0, 3, 1, 2)  # torch.Size([1, 128, 200, 322])

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

        x = self.Locally_Recurrent(x)  # torch.Size([1, 64, 200, 322])

        x = x.permute(0, 3, 1, 2).reshape(B*S, N, K)

        x = self.Conv1D(x)

        x = x.reshape(B, S, N, -1).permute(0, 2, 3, 1)

        x = self.Globally_Attentive(x)

        x = x.permute(0, 3, 1, 2).reshape(B*S, N, -1)

        x = self.ConvTrans1D(x)

        x = x.reshape(B, S, N, -1).permute(0, 2, 3, 1)

        return x


class Separation(nn.Module):

    def __init__(self, in_channels, out_channels, length, Cycle, hidden_channels, bidirectional, num_heads, Spk):

        super(Separation, self).__init__()

        self.LayerNorm = nn.LayerNorm(normalized_shape=in_channels)

        self.Linear = nn.Linear(in_features=in_channels, out_features=out_channels)

        self.Segmentation = Segmentation(length=length)

        self.Cycle = Cycle

        kernel_size = []
        stride = []

        for i in range(self.Cycle):
            if i < self.Cycle//2:
                kernel_size.append(4**i)
                stride.append(4**i)
            else:
                kernel_size.append(4**(self.Cycle-i-1))
                stride.append(4**(self.Cycle-i-1))

        self.Sandglasset_Blocks = nn.ModuleList([])

        for i in range(self.Cycle):
            self.Sandglasset_Blocks.append(Sandglasset_Block(in_channels=out_channels,
                                                             hidden_channels=hidden_channels,
                                                             bidirectional=bidirectional,
                                                             num_heads=num_heads,
                                                             kernel_size=kernel_size[i],
                                                             stride=stride[i]))

        self.residual = []

        self.PReLU = nn.PReLU()

        self.Spk = Spk

        self.Conv2d = nn.Conv2d(in_channels=out_channels,
                                out_channels=Spk*in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.LayerNorm(x.permute(0, 2, 1))

        x = self.Linear(x).permute(0, 2, 1)

        x, gap = self.Segmentation(x)

        self.residual = []

        for i in range(self.Cycle):

            x = self.Sandglasset_Blocks[i](x)

            if i < self.Cycle//2:
                self.residual.append(x)
            else:
                x = x + self.residual[self.Cycle-i-1]

        x = self.PReLU(x)

        x = self.Conv2d(x)

        B, _, K, S = x.shape

        x = x.reshape(B*self.Spk, -1, K, S)  # torch.Size([4, 256, 256, 128])

        x = self._over_add(x, gap)  # torch.Size([2, 256, 16002])

        _, N, L = x.shape

        x = x.reshape(B, self.Spk, N, L)  # torch.Size([1, 2, 128, 31999])

        x = self.ReLU(x)

        x = x.transpose(0, 1)  # torch.Size([2, 1, 128, 31999])

        return x

    def _over_add(self, x, gap):
        """
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """

        B, N, K, S = x.shape

        P = K // 2

        # [B, N, S, K]
        x = x.transpose(2, 3).reshape(B, N, -1, K * 2)

        x1 = x[:, :, :, :K].reshape(B, N, -1)[:, :, P:]
        x2 = x[:, :, :, K:].reshape(B, N, -1)[:, :, :-P]
        x = x1 + x2

        # [B, N, L]
        if gap > 0:
            x = x[:, :, :-gap]

        return x


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(Decoder, self).__init__()

        if kernel_size == 1:
            self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0)
        else:
            self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=kernel_size//2,
                                                      padding=0)

    def forward(self, x):

        x = self.ConvTranspose1d(x)

        return x


class Sandglasset(nn.Module):

    def __init__(self, M=4, E=256, kernel_size=4, D=128, K=256, N=6, H=128, bidirectional=True, J=8, C=2):

        super(Sandglasset, self).__init__()

        self.M = M
        self.E = E
        self.kernel_size = kernel_size
        self.D = D
        self.K = K
        self.N = N
        self.H = H
        self.bidirectional = bidirectional
        self.J = J
        self.C = C

        self.framing = framing(length=self.M)

        self.Encoder = Encoder(in_channels=self.M, out_channels=self.E, kernel_size=self.kernel_size)

        self.Separation = Separation(in_channels=self.E, out_channels=self.D, length=self.K, Cycle=self.N,
                                     hidden_channels=self.H, bidirectional=bidirectional, num_heads=self.J,
                                     Spk=self.C)

        self.Decoder = Decoder(in_channels=self.E, out_channels=self.M, kernel_size=self.kernel_size)

    def forward(self, x):

        x, gap = self.framing(x)  # torch.Size([2, 4, 16002])

        e = self.Encoder(x)  # torch.Size([2, 256, 8000])

        s = self.Separation(e)  # torch.Size([4, 256, 8000])

        out = [s[i]*e for i in range(self.C)]

        del e, s

        audio = [self.Decoder(out[i]) for i in range(self.C)]

        audio = [self._over_add(audio[i], gap) for i in range(self.C)]

        audio = torch.cat(audio, dim=1)  # [B, C, T]

        return audio

    def _over_add(self, x, gap):
        """
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """

        B, K, S = x.shape

        P = K // 2

        # [B, N, S, K]
        x = x.transpose(1, 2).reshape(B, -1, K * 2)

        x1 = x[:, :, :K].reshape(B, -1)[:, P:]
        x2 = x[:, :, K:].reshape(B, -1)[:, :-P]
        x = x1 + x2

        # [B, N, L]
        if gap > 0:
            x = x[:, :-gap]

        x = torch.unsqueeze(input=x, dim=1)

        return x

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(M=package['M'], E=package['E'], kernel_size=package['kernel_size'],
                    D=package['D'], K=package['K'], N=package['N'], H=package['H'],
                    bidirectional=package['bidirectional'], J=package['J'], C=package['C'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'M': model.M, 'E': model.E, 'kernel_size': model.kernel_size,
            'D': model.D, 'K': model.K, 'N': model.N, 'H': model.H,
            'bidirectional': model.bidirectional, 'J': model.J, 'C': model.C,

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

    x = torch.rand(2, 32000)

    model = Sandglasset(M=4,
                        E=256,
                        kernel_size=4,
                        D=128,
                        K=256,
                        N=2,
                        H=128,
                        bidirectional=True,
                        J=8,
                        C=2)

    y = model(x)

    print(y.shape)
