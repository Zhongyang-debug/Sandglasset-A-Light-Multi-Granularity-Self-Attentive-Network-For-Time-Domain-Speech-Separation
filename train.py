import argparse
import torch
from dataset.data import AudioDataLoader, AudioDataset
from src.trainer import Trainer
from model.conv_tasnet import ConvTasNet
from model.dual_path_rnn import Dual_RNN_model
from model.dptnet import DPTNet
from model.sepformer import Sepformer
from model.sudormrf import SuDORMRF
from model.galr import GALR
from model.sandglasset import Sandglasset
import json5
import numpy as np
from adamp import AdamP, SGDP


def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 数据
    tr_dataset = AudioDataset(json_dir=config["train_dataset"]["train_dir"],  # 目录下包含 mix.json, s1.json, s2.json
                              batch_size=config["train_dataset"]["batch_size"],
                              sample_rate=config["train_dataset"]["sample_rate"],  # 采样率
                              segment=config["train_dataset"]["segment"])  # 语音时长

    cv_dataset = AudioDataset(json_dir=config["validation_dataset"]["validation_dir"],
                              batch_size=config["validation_dataset"]["batch_size"],
                              sample_rate=config["validation_dataset"]["sample_rate"],
                              segment=config["validation_dataset"]["segment"],
                              cv_max_len=config["validation_dataset"]["cv_max_len"])

    tr_loader = AudioDataLoader(tr_dataset,
                                batch_size=config["train_loader"]["batch_size"],
                                shuffle=config["train_loader"]["shuffle"],
                                num_workers=config["train_loader"]["num_workers"])

    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size=config["validation_loader"]["batch_size"],
                                shuffle=config["validation_loader"]["shuffle"],
                                num_workers=config["validation_loader"]["num_workers"])

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # 模型
    if config["model"]["type"] == "sandglasset":
        model = Sandglasset(in_channels=config["model"]["sandglasset"]["in_channels"],
                            out_channels=config["model"]["sandglasset"]["out_channels"],
                            kernel_size=config["model"]["galr"]["kernel_size"],
                            length=config["model"]["sandglasset"]["length"],
                            hidden_channels=config["model"]["sandglasset"]["hidden_channels"],
                            num_layers=config["model"]["sandglasset"]["num_layers"],
                            bidirectional=config["model"]["sandglasset"]["bidirectional"],
                            num_heads=config["model"]["sandglasset"]["num_heads"],
                            cycle_amount=config["model"]["sandglasset"]["cycle_amount"],
                            speakers=config["model"]["sandglasset"]["speakers"])
    else:
        print("No loaded model!")

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"], config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
    else:
        print("Not support optimizer")
        return

    trainer = Trainer(data, model, optimize, config)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/train/train.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)
