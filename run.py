import argparse
import os
import pickle
from abc import ABC, abstractmethod
import time

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import nn
from torch_geometric.nn.conv import WLConvContinuous
from torch_geometric.nn.models import GAT, GCN, GIN, MLP, LabelPropagation

from logger import Logger
from models import SG, MixHop, MixSG, WLContinous


class BaseRunner(ABC):
    def __init__(self, args):
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

        if args.undirected == "true":
            self.dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=T.ToUndirected())
        elif args.undirected == "false":
            self.dataset = PygNodePropPredDataset(name="ogbn-arxiv")
        else:
            raise ValueError("Invalid undirected")

        self.data = self.dataset[0].to(self.device)
        self.split_idx = self.dataset.get_idx_split()
        self.train_idx = self.split_idx["train"].to(self.device)
        self.evaluator = Evaluator(name="ogbn-arxiv")

        self.conv_model: nn.Module

    def test(self, y_pred):
        train_acc = self.evaluator.eval(
            {
                "y_true": self.data.y[self.split_idx["train"]],
                "y_pred": y_pred[self.split_idx["train"]],
            }
        )["acc"]
        valid_acc = self.evaluator.eval(
            {
                "y_true": self.data.y[self.split_idx["valid"]],
                "y_pred": y_pred[self.split_idx["valid"]],
            }
        )["acc"]
        test_acc = self.evaluator.eval(
            {
                "y_true": self.data.y[self.split_idx["test"]],
                "y_pred": y_pred[self.split_idx["test"]],
            }
        )["acc"]

        return train_acc, valid_acc, test_acc

    @abstractmethod
    def run(self):
        pass


class LabelRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)

        self.conv_model = LabelPropagation(num_layers=args.conv_num_layers, alpha=args.alpha).to(self.device)

    def predict(self):
        out = self.conv_model(y=self.data.y, edge_index=self.data.edge_index, mask=self.train_idx)
        y_pred = out.argmax(dim=-1, keepdim=True)

        return y_pred

    def run(self):
        y_pred = self.predict()
        result = self.test(y_pred)
        logger.add_result(run, result)


class WLRunner(BaseRunner):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epochs = args.epochs
        self.lr = args.lr
        self.conv_model = WLContinous(data=self.data, num_layers=args.conv_num_layers, cached=True).to(self.device)

        clf_channels_list = [self.data.x.size(-1)]
        for _ in range(args.clf_num_layers - 2):
            clf_channels_list.append(clf_channels_list[-1] // 2)
        clf_channels_list.append(self.dataset.num_classes)
        self.clf = MLP(channel_list=clf_channels_list, cached=True).to(self.device)
        
    def learn(self, criterion, optimizer):
        self.clf.train()

        hidden = self.conv_model(x=self.data.x, edge_index=self.data.edge_index)[self.train_idx]
        out = self.clf(hidden)
        loss = criterion(out, self.data.y.squeeze(1)[self.train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self):
        self.clf.eval()

        hidden = self.conv_model(x=self.data.x, edge_index=self.data.edge_index)
        out = self.clf(hidden)
        y_pred = out.argmax(dim=-1, keepdim=True)

        return y_pred

    def run(self):
        self.clf.reset_parameters()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.clf.parameters(), self.lr)

        for epoch in range(self.epochs):
            loss = self.learn(criterion=criterion, optimizer=optimizer)
            y_pred = self.predict()
            result = self.test(y_pred)

            logger.add_result(run, result)

            train_acc, valid_acc, test_acc = result
            print(
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * train_acc:.2f}%, "
                f"Valid: {100 * valid_acc:.2f}% "
                f"Test: {100 * test_acc:.2f}%"
            )


class GNNRunner(BaseRunner):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epochs = args.epochs
        self.lr = args.lr

        if args.conv == "gcn":
            self.conv_model = GCN(
                in_channels=self.data.num_features,
                hidden_channels=args.conv_hidden_channels,
                num_layers=args.conv_num_layers,
                cached=True,
            )
        elif args.conv == "gat":
            self.conv_model = GAT(
                in_channels=self.data.num_features,
                hidden_channels=args.conv_hidden_channels,
                num_layers=args.conv_num_layers,
                cached=True,
            )
        elif args.conv == "gin":
            self.conv_model = GIN(
                in_channels=self.data.num_features,
                hidden_channels=args.conv_hidden_channels,
                num_layers=args.conv_num_layers,
                cached=True,
            )
        elif args.conv == "sgc":
            self.conv_model = SG(
                in_channels=self.data.num_features,
                out_channels=args.conv_hidden_channels,
                hops=args.conv_hops,
                cached=True,
            )
        elif args.conv == "mixhop":
            self.conv_model = MixHop(
                in_channels=self.data.num_features,
                out_channels=args.conv_hidden_channels // args.conv_hops,
                hops=args.conv_hops,
                num_layers=args.conv_num_layers,
                cached=True,
            )
        elif args.conv == "mixsgc":
            self.conv_model = MixSG(
                data=self.data,
                out_channels=args.conv_hidden_channels,
                hops=args.conv_hops,
                cached=True,
            )
        else:
            raise ValueError("Invalid convolution")

        self.conv_model = self.conv_model.to(self.device)

        clf_channels_list = [args.conv_hidden_channels]
        for _ in range(args.clf_num_layers - 2):
            clf_channels_list.append(clf_channels_list[-1] // 2)
        clf_channels_list.append(self.dataset.num_classes)
        self.clf = MLP(channel_list=clf_channels_list, cached=True).to(self.device)
        
    def learn(self, criterion, optimizer):
        self.conv_model.train()
        self.clf.train()

        hidden = self.conv_model(x=self.data.x, edge_index=self.data.edge_index)[self.train_idx]
        out = self.clf(hidden)
        loss = criterion(out, self.data.y.squeeze(1)[self.train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self):
        with torch.no_grad():
            self.conv_model.eval()

            hidden = self.conv_model(x=self.data.x, edge_index=self.data.edge_index)
            out = self.clf(hidden)
            y_pred = out.argmax(dim=-1, keepdim=True)

            return y_pred

    def run(self):
        self.clf.reset_parameters()
        self.conv_model.reset_parameters()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(list(self.conv_model.parameters()) + list(self.clf.parameters()), self.lr)

        for epoch in range(self.epochs):
            loss = self.learn(criterion=criterion, optimizer=optimizer)
            y_pred = self.predict()
            result = self.test(y_pred)

            logger.add_result(run, result)

            train_acc, valid_acc, test_acc = result
            print(
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * train_acc:.2f}%, "
                f"Valid: {100 * valid_acc:.2f}% "
                f"Test: {100 * test_acc:.2f}%"
            )


if __name__ == "__main__":
    # Parse args and initialize logger
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (GNN)")
    parser.add_argument("--undirected", type=str, choices=["true", "false"], default="true")
    parser.add_argument(
        "--conv",
        type=str,
        choices=[
            "lp",
            "wl",
            "sgc",
            "mixsgc",
            "gcn",
            "gin",
            "mixhop",
        ],
        default="mixsgc",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conv_num_layers", type=int, default=3)
    parser.add_argument("--conv_hidden_channels", type=int, default=63)  # For mixhop, hidden channels needs to be divisible by hops
    parser.add_argument("--conv_hops", type=int, default=3)
    parser.add_argument("--clf_num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    print(args)
    logger = Logger(args.runs, args)

    # Make sure that hops divides hidden channels
    args.conv_hidden_channels = (args.conv_hidden_channels // args.conv_hops) * args.conv_hops

    # Initialize runner
    if args.conv in ["lp"]:
        runner = LabelRunner(args=args)
    elif args.conv in ["wl"]:
        runner = WLRunner(args=args)
    else:
        runner = GNNRunner(args=args)
    print(runner.conv_model)

    # Begin training
    for run in range(args.runs):
        runner.run()
        logger.print_statistics(run)
    logger.print_statistics()

    os.makedirs("logs", exist_ok=True)

    with open(f'logs/{args.conv}.pkl', 'wb') as f:
        pickle.dump(logger, f)