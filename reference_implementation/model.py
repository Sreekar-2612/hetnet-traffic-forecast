import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGNNEncoder(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=64):
        super().__init__()
        # NOVELTY: each node type and edge type gets its own conv weights
        self.conv1 = HeteroConv({
            ('macro','geo','macro'):   SAGEConv(in_dim, hidden),
            ('pico','geo','pico'):     SAGEConv(in_dim, hidden),
            ('femto','geo','femto'):   SAGEConv(in_dim, hidden),
            ('macro','cross','pico'):  SAGEConv(in_dim, hidden),
            ('macro','cross','femto'): SAGEConv(in_dim, hidden),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('macro','geo','macro'):   SAGEConv(hidden, out_dim),
            ('pico','geo','pico'):     SAGEConv(hidden, out_dim),
            ('femto','geo','femto'):   SAGEConv(hidden, out_dim),
            ('macro','cross','pico'):  SAGEConv(hidden, out_dim),
            ('macro','cross','femto'): SAGEConv(hidden, out_dim),
        }, aggr='sum')
        
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.ModuleDict({
            'macro': nn.Linear(out_dim, out_dim),
            'pico':  nn.Linear(out_dim, out_dim),
            'femto': nn.Linear(out_dim, out_dim),
        })

    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        out = self.conv1(x_dict, edge_index_dict)
        out = {k: F.relu(v) for k, v in out.items()}
        
        # Layer 2
        out = self.conv2(out, edge_index_dict)
        # Apply layer norm and relu
        out = {k: F.relu(self.norm(v)) for k, v in out.items()}
        
        # Projection
        return {k: self.proj[k](v) for k, v in out.items()}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1)) # (max_len, 1, d_model)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TASTF(nn.Module):
    def __init__(self, N, gnn_out=64, nhead=4, tf_layers=2, horizon=3,
                 macro_idx=None, pico_idx=None, femto_idx=None):
        super().__init__()
        self.N, self.horizon = N, horizon
        self.macro_idx, self.pico_idx, self.femto_idx = macro_idx, pico_idx, femto_idx
        
        self.gnn = HeteroGNNEncoder(in_dim=1, hidden=32, out_dim=gnn_out)
        self.pos_enc = PositionalEncoding(gnn_out)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=gnn_out, nhead=nhead,
                                               dim_feedforward=128, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        
        self.head = nn.Linear(gnn_out, horizon)

    def forward(self, x_seq, hetero_graph):
        """
        x_seq: (B, T, N)
        hetero_graph: HeteroData object
        """
        B, T, N = x_seq.shape
        device = x_seq.device
        
        # 1. Spatial encoding per timestep
        spatial_seq = []
        for t in range(T):
            xt = x_seq[:, t, :] # (B, N)
            
            # Map global cell IDs to tier-local nodes
            # Reshape to (B*N_tier, 1) for GNN
            x_dict = {
                'macro': xt[:, self.macro_idx].reshape(-1, 1),
                'pico':  xt[:, self.pico_idx].reshape(-1, 1),
                'femto': xt[:, self.femto_idx].reshape(-1, 1),
            }
            
            emb_dict = self.gnn(x_dict, hetero_graph.edge_index_dict)
            
            # Reconstruct full spatial map (B, N, hidden)
            # emb_dict['macro'] is (B*N_macro, hidden)
            full = torch.zeros(B, N, 64, device=device)
            full[:, self.macro_idx] = emb_dict['macro'].view(B, -1, 64)
            full[:, self.pico_idx]  = emb_dict['pico'].view(B, -1, 64)
            full[:, self.femto_idx] = emb_dict['femto'].view(B, -1, 64)
            
            spatial_seq.append(full)
            
        # 2. Temporal encoding
        # seq: (T, B*N, hidden)
        seq = torch.stack(spatial_seq, dim=0).view(T, B*N, 64)
        seq = self.pos_enc(seq)
        
        # Transformer processing
        out = self.transformer(seq)
        
        # 3. Last timestep prediction
        final_emb = out[-1] # (B*N, hidden)
        
        # (B*N, horizon) -> (B, N, horizon) -> (B, horizon, N)
        pred = self.head(final_emb).view(B, N, self.horizon).permute(0, 2, 1)
        
        return pred

if __name__ == "__main__":
    # Smoke test
    N = 100
    m, p, f = range(0,10), range(10, 70), range(70, 100)
    model = TASTF(N, macro_idx=list(m), pico_idx=list(p), femto_idx=list(f))
    x = torch.randn(32, 12, N)
    
    from graph_builder import build_hetero_graph
    import numpy as np
    hetero, _, _, _ = build_hetero_graph(np.random.rand(100, N))
    
    out = model(x, hetero)
    print(f"Output shape: {out.shape}") # Expected: (32, 3, 100)
