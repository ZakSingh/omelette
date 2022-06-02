import argparse
import copy
from PropLang import PropLang

from pytorch_lightning.plugins import SingleDevicePlugin, DDPSpawnPlugin

from ppo import PPOAgent
from rejoice.pretrain_dataset_gen import generate_dataset
import torch_geometric as pyg
from rejoice.PretrainingDataset import PretrainingDataset
from rejoice.tests.test_lang import TestLang
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import os
import torchmetrics
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_node_features", type=int, default=7,
                        help="the number of node features for input graphs")
    parser.add_argument("--num_actions", type=int, default=7,
                        help="the number of node features for input graphs")
    parser.add_argument("--generate", type=bool, default=False,
                        help="the number of node features for input graphs")
    parser.add_argument("--count", type=int, default=100_000,
                        help="the number of expressions to generate")
    parser.add_argument("-data_root", type=str, default="./PropLang",
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
    def __init__(self, model, batch_size: int):
        super().__init__()
        # self.save_hyperparameters()
        self.batch_size = batch_size
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
        self.log("train_loss", loss, batch_size=self.batch_size)
        self.log("acc/train_acc", acc, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("acc/val_acc", acc, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("acc/test_acc", acc, batch_size=self.batch_size)


def main():
    args = parse_args()
    lang = PropLang()

    if args.generate:
        generate_dataset(lang, num=args.count)

    envs_mock = Struct(**{
        "single_observation_space": Struct(**{
            "num_node_features": lang.num_node_features  # args.num_node_features
        }),
        "single_action_space": Struct(**{
            "n": lang.num_rules + 1  # args.num_actions
        })
    })

    batch_size = 32

    agent = PPOAgent(envs=envs_mock)
    model = PretrainModule(copy.deepcopy(agent.actor), batch_size=batch_size)
    dataset = PretrainingDataset(lang=lang, root=args.data_root)
    train_data, val_data, test_data = split_dataset(dataset)

    pyg_lightning_dataset = pyg.data.LightningDataset(train_dataset=train_data,
                                                      val_dataset=val_data,
                                                      test_dataset=test_data,
                                                      batch_size=batch_size,
                                                      num_workers=4
                                                      )

    trainer = pl.Trainer(strategy=DDPSpawnPlugin(find_unused_parameters=False),
                         log_every_n_steps=5,
                         accelerator='gpu',
                         devices=1,
                         max_epochs=5_000)
    trainer.fit(model, pyg_lightning_dataset)
    torch.save(model.model.state_dict(), "./weights.pt")


if __name__ == "__main__":
    main()
