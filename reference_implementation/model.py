import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


def batch_edge_index(edge_index: Tensor, batch_size: int, num_nodes: int) -> Tensor:
    rows, cols = [], []
    for b in range(batch_size):
        rows.append(edge_index[0] + b * num_nodes)
        cols.append(edge_index[1] + b * num_nodes)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


class HeteroGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        hidden: int = 32,
        out_dim: int = 64,
        use_gat: bool = False,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_gat = use_gat
        self.out_dim = out_dim
        if use_gat:
            h1, h2 = hidden // heads, out_dim // heads
            conv1 = {
                ("macro", "geo", "macro"): GATConv(
                    in_dim, h1, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("pico", "geo", "pico"): GATConv(
                    in_dim, h1, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("femto", "geo", "femto"): GATConv(
                    in_dim, h1, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("macro", "cross", "pico"): GATConv(
                    in_dim, h1, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("macro", "cross", "femto"): GATConv(
                    in_dim, h1, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
            }
            conv2 = {
                ("macro", "geo", "macro"): GATConv(
                    hidden, h2, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("pico", "geo", "pico"): GATConv(
                    hidden, h2, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("femto", "geo", "femto"): GATConv(
                    hidden, h2, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("macro", "cross", "pico"): GATConv(
                    hidden, h2, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
                ("macro", "cross", "femto"): GATConv(
                    hidden, h2, heads=heads, concat=True, dropout=dropout, add_self_loops=False
                ),
            }
        else:
            conv1 = {
                ("macro", "geo", "macro"): SAGEConv(in_dim, hidden),
                ("pico", "geo", "pico"): SAGEConv(in_dim, hidden),
                ("femto", "geo", "femto"): SAGEConv(in_dim, hidden),
                ("macro", "cross", "pico"): SAGEConv(in_dim, hidden),
                ("macro", "cross", "femto"): SAGEConv(in_dim, hidden),
            }
            conv2 = {
                ("macro", "geo", "macro"): SAGEConv(hidden, out_dim),
                ("pico", "geo", "pico"): SAGEConv(hidden, out_dim),
                ("femto", "geo", "femto"): SAGEConv(hidden, out_dim),
                ("macro", "cross", "pico"): SAGEConv(hidden, out_dim),
                ("macro", "cross", "femto"): SAGEConv(hidden, out_dim),
            }

        self.conv1 = HeteroConv(conv1, aggr="sum")
        self.conv2 = HeteroConv(conv2, aggr="sum")
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.ModuleDict(
            {
                "macro": nn.Linear(out_dim, out_dim),
                "pico": nn.Linear(out_dim, out_dim),
                "femto": nn.Linear(out_dim, out_dim),
            }
        )

    def forward(self, x_dict, edge_index_dict):
        out = self.conv1(x_dict, edge_index_dict)
        out = {k: F.relu(v) for k, v in out.items()}
        out = self.conv2(out, edge_index_dict)
        out = {k: F.relu(self.norm(v)) for k, v in out.items()}
        return {k: self.proj[k](v) for k, v in out.items()}


class HomoGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        use_gat: bool = True,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        if use_gat:
            h1, h2 = hidden // heads, out_dim // heads
            self.conv1 = GATConv(in_dim, h1, heads=heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden, h2, heads=heads, concat=True, dropout=dropout)
        else:
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return F.relu(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(0)]


class TASTF(nn.Module):
    def __init__(
        self,
        N: int,
        gnn_out: int = 64,
        nhead: int = 4,
        tf_layers: int = 2,
        horizon: int = 3,
        macro_idx=None,
        pico_idx=None,
        femto_idx=None,
        use_gat: bool = False,
        temporal_dim: int = 0,
        ff_dim: int = 128,
        dropout: float = 0.1,
        probabilistic: bool = False,
    ):
        super().__init__()
        self.N = N
        self.horizon = horizon
        self.gnn_out = gnn_out
        self.macro_idx = macro_idx
        self.pico_idx = pico_idx
        self.femto_idx = femto_idx
        self.temporal_dim = temporal_dim
        self.probabilistic = probabilistic

        in_dim = 1 + temporal_dim
        self.gnn = HeteroGNNEncoder(
            in_dim=in_dim,
            hidden=32,
            out_dim=gnn_out,
            use_gat=use_gat,
            heads=4,
            dropout=dropout,
        )
        self.pos_enc = PositionalEncoding(gnn_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gnn_out,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=False,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)

        if probabilistic:
            self.head_mu = nn.Linear(gnn_out, horizon)
            self.head_logvar = nn.Linear(gnn_out, horizon)
        else:
            self.head = nn.Linear(gnn_out, horizon)

    def forward(self, x_seq: Tensor, hetero_graph: HeteroData, time_feat: Tensor | None = None):
        B, T, N = x_seq.shape
        device = x_seq.device

        if self.temporal_dim > 0:
            if time_feat is None:
                raise ValueError("time_feat required when temporal_dim > 0")
            tf = time_feat.unsqueeze(1).expand(B, T, self.temporal_dim)
        else:
            tf = None

        spatial_seq = []
        for t in range(T):
            xt = x_seq[:, t, :]
            if tf is not None:
                tf_t = tf[:, t, :].unsqueeze(1).expand(B, N, self.temporal_dim)
                x_full = torch.cat([xt.unsqueeze(-1), tf_t], dim=-1)
            else:
                x_full = xt.unsqueeze(-1)

            x_dict = {
                "macro": x_full[:, self.macro_idx, :].reshape(-1, x_full.size(-1)),
                "pico": x_full[:, self.pico_idx, :].reshape(-1, x_full.size(-1)),
                "femto": x_full[:, self.femto_idx, :].reshape(-1, x_full.size(-1)),
            }

            emb_dict = self.gnn(x_dict, hetero_graph.edge_index_dict)

            full = torch.zeros(B, N, self.gnn_out, device=device)
            full[:, self.macro_idx] = emb_dict["macro"].view(B, -1, self.gnn_out)
            full[:, self.pico_idx] = emb_dict["pico"].view(B, -1, self.gnn_out)
            full[:, self.femto_idx] = emb_dict["femto"].view(B, -1, self.gnn_out)
            spatial_seq.append(full)

        seq = torch.stack(spatial_seq, dim=0).view(T, B * N, self.gnn_out)
        seq = self.pos_enc(seq)
        out = self.transformer(seq)
        final_emb = out[-1]

        if self.probabilistic:
            mu = self.head_mu(final_emb).view(B, N, self.horizon)
            logv = self.head_logvar(final_emb).view(B, N, self.horizon)
            logv = logv.clamp(-10, 10)
            return mu.permute(0, 2, 1), logv.permute(0, 2, 1)
        pred = self.head(final_emb).view(B, N, self.horizon).permute(0, 2, 1)
        return pred


class TASTFHomo(nn.Module):
    """Homogeneous graph ablation: single node type, same transformer head."""

    def __init__(
        self,
        N: int,
        gnn_out: int = 64,
        nhead: int = 4,
        tf_layers: int = 2,
        horizon: int = 3,
        use_gat: bool = True,
        temporal_dim: int = 0,
        ff_dim: int = 128,
        dropout: float = 0.1,
        probabilistic: bool = False,
    ):
        super().__init__()
        self.N = N
        self.horizon = horizon
        self.gnn_out = gnn_out
        self.temporal_dim = temporal_dim
        self.probabilistic = probabilistic
        in_dim = 1 + temporal_dim
        self.gnn = HomoGNNEncoder(in_dim, hidden=32, out_dim=gnn_out, use_gat=use_gat, dropout=dropout)
        self.pos_enc = PositionalEncoding(gnn_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gnn_out,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=False,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        if probabilistic:
            self.head_mu = nn.Linear(gnn_out, horizon)
            self.head_logvar = nn.Linear(gnn_out, horizon)
        else:
            self.head = nn.Linear(gnn_out, horizon)

    def forward(self, x_seq: Tensor, graph: Data, time_feat: Tensor | None = None):
        B, T, N = x_seq.shape
        device = x_seq.device
        edge_index = graph.edge_index.to(device)

        if self.temporal_dim > 0:
            if time_feat is None:
                raise ValueError("time_feat required when temporal_dim > 0")
            tf = time_feat.unsqueeze(1).expand(B, T, self.temporal_dim)
        else:
            tf = None

        spatial_seq = []
        ei_b = batch_edge_index(edge_index, B, N)
        for t in range(T):
            xt = x_seq[:, t, :]
            if tf is not None:
                tf_t = tf[:, t, :].unsqueeze(1).expand(B, N, self.temporal_dim)
                x_full = torch.cat([xt.unsqueeze(-1), tf_t], dim=-1)
            else:
                x_full = xt.unsqueeze(-1)
            x_flat = x_full.reshape(B * N, -1)
            h = self.gnn(x_flat, ei_b)
            spatial_seq.append(h.view(B, N, self.gnn_out))

        seq = torch.stack(spatial_seq, dim=0).view(T, B * N, self.gnn_out)
        seq = self.pos_enc(seq)
        out = self.transformer(seq)
        final_emb = out[-1]

        if self.probabilistic:
            mu = self.head_mu(final_emb).view(B, N, self.horizon)
            logv = self.head_logvar(final_emb).view(B, N, self.horizon)
            logv = logv.clamp(-10, 10)
            return mu.permute(0, 2, 1), logv.permute(0, 2, 1)
        pred = self.head(final_emb).view(B, N, self.horizon).permute(0, 2, 1)
        return pred


def gaussian_nll(pred_mu: Tensor, pred_logvar: Tensor, target: Tensor) -> Tensor:
    """Mean over batch; per-element NLL under diagonal Gaussian."""
    inv_var = torch.exp(-pred_logvar)
    return 0.5 * ((target - pred_mu) ** 2 * inv_var + pred_logvar + math.log(2 * math.pi)).mean()
