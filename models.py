import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.nn.conv import (MixHopConv, SGConv, SimpleConv,
                                     WLConvContinuous)
from torch_geometric.nn.models import MLP


class WLContinous(nn.Module):
    def __init__(self, data, num_layers: int, cached: bool = True) -> None:
        super().__init__()

        convs = [WLConvContinuous(cached=cached) for _ in range(num_layers)]
        self.convs = nn.ModuleList(convs)

        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index)
        self.x = x
        
    def forward(self, **kwargs) -> torch.Tensor:
        return self.x


class SG(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hops: int, cached: bool = True) -> None:
        super().__init__()

        self.conv = SGConv(in_channels=in_channels, out_channels=out_channels, K=hops, cached=cached)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)

    def reset_parameters(self):
        self.conv.reset_parameters()


class MixHop(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hops: int, num_layers: int, cached: bool = True) -> None:
        super().__init__()

        powers = [k for k in range(hops)]

        convs = [MixHopConv(in_channels=in_channels, out_channels=out_channels, powers=powers, cached=cached)]
        for _ in range(num_layers - 1):
            convs.append(MixHopConv(in_channels=len(powers) * out_channels, out_channels=out_channels, powers=powers, cached=cached))
        self.convs = nn.ModuleList(convs)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.relu(conv(x, edge_index))

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class MixSG(nn.Module):
    def __init__(self, data, out_channels: int, hops: int, cached: bool = True) -> None:
        super().__init__()

        self.conv = SimpleConv(aggr="mean", combine_root=None, cached=cached)

        x_lst = [data.x]
        sum_x = torch.zeros_like(data.x)
        for _ in range(hops):
            new_x = self.conv.forward(x=x_lst[-1], edge_index=data.edge_index)

            new_x -= sum_x
            sum_x += new_x
            x_lst.append(new_x)

        self.x = torch.concat(x_lst, dim=1)
        self.w = nn.Linear(self.x.size(-1), out_channels)

    def forward(self, **kwargs) -> torch.Tensor:
        return self.w(self.x)

    def reset_parameters(self):
        self.w.reset_parameters()
