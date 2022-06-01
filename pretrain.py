import argparse
import copy

from pytorch_lightning.plugins import SingleDevicePlugin

from ppo import PPOAgent
import torch_geometric as pyg
from rejoice.PretrainingDataset import PretrainingDataset
from rejoice.tests.test_lang import TestLang
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import os
import torchmetrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_node_features", type=int, default=7,
                        help="the number of node features for input graphs")
    parser.add_argument("--num_actions", type=int, default=7,
                        help="the number of node features for input graphs")
    parser.add_argument("-data_root", type=str, default="./rejoice/TestLang",
                        help="Root directory for data")
    args = parser.parse_args()
    return args


def split_dataset(dataset, train=0.6, val=0.2, test=0.2):
    t = int(train * len(dataset))
    v = int((train + val) * len(dataset))
    train_dataset = dataset[:t]
    val_dataset = dataset[t:v]
    test_dataset = dataset[v:]
    return train_dataset, val_dataset, test_dataset


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class PretrainModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.accuracy = torchmetrics.Accuracy()
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):
        x = self.model(data)
        loss = self.loss_module(input=x, target=data.y)
        acc = self.accuracy(x, data.y)
        return loss, acc

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("test_acc", acc)


def main():
    args = parse_args()
    lang = TestLang()
    envs_mock = Struct(**{
        "single_observation_space": Struct(**{
            "num_node_features": lang.num_node_features  # args.num_node_features
        }),
        "single_action_space": Struct(**{
            "n": lang.num_actions + 1  # args.num_actions
        })
    })

    agent = PPOAgent(envs=envs_mock)
    model = PretrainModule(copy.deepcopy(agent.actor))
    dataset = PretrainingDataset(lang=lang, root=args.data_root)
    train_data, val_data, test_data = split_dataset(dataset)

    pyg_lightning_dataset = pyg.data.LightningDataset(train_dataset=train_data,
                                                      val_dataset=val_data,
                                                      test_dataset=test_data,
                                                      # num_workers=6
                                                      )

    trainer = pl.Trainer(strategy=SingleDevicePlugin())
    trainer.fit(model, pyg_lightning_dataset)


if __name__ == "__main__":
    main()
