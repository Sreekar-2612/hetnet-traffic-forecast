import torch
import numpy as np
from torch_geometric.data import HeteroData

def build_hetero_graph(data_matrix, k=5, macro_pct=0.10, femto_pct=0.30):
    """
    Constructs a heterogeneous graph from traffic activity.
    data_matrix: (T, N) where T is time and N is number of cells.
    """
    T, N = data_matrix.shape
    mean_act = data_matrix.mean(axis=0)
    sorted_idx = np.argsort(mean_act)
    
    n_femto = int(N * femto_pct)
    n_macro = int(N * macro_pct)
    
    femto_idx = sorted_idx[:n_femto]
    macro_idx = sorted_idx[-n_macro:]
    pico_idx  = sorted_idx[n_femto:-n_macro]
    
    # Local mapping (grid_id -> index within tier)
    macro_local = {g:l for l,g in enumerate(macro_idx)}
    pico_local  = {g:l for l,g in enumerate(pico_idx)}
    femto_local = {g:l for l,g in enumerate(femto_idx)}
    
    hetero = HeteroData()
    
    # Features: use mean activity as a static feature (or could use current activity)
    hetero['macro'].x = torch.tensor(mean_act[macro_idx]).unsqueeze(1).float()
    hetero['pico'].x  = torch.tensor(mean_act[pico_idx]).unsqueeze(1).float()
    hetero['femto'].x = torch.tensor(mean_act[femto_idx]).unsqueeze(1).float()
    
    # Assume grid layout for coordinates for KNN
    sqrt_n = int(np.sqrt(N))
    coords = np.array([[i//sqrt_n, i%sqrt_n] for i in range(N)], dtype=np.float32)
    
    def knn_edges(src_ids, dst_ids, k):
        src_e, dst_e = [], []
        # local_maps stores (global_grid_id -> tier_local_id)
        # However, dst edges should be local to their respective tier in HeteroData
        
        # We need identifying which tier mapping to use based on dst_ids
        if any(g in macro_local for g in dst_ids): local_map_dst = macro_local
        elif any(g in pico_local for g in dst_ids): local_map_dst = pico_local
        else: local_map_dst = femto_local
        
        if any(g in macro_local for g in src_ids): local_map_src = macro_local
        elif any(g in pico_local for g in src_ids): local_map_src = pico_local
        else: local_map_src = femto_local

        for s_g in src_ids:
            # Calculate distances to all global IDs in dst_ids
            dists = np.linalg.norm(coords[dst_ids] - coords[s_g], axis=1)
            # Find K nearest neighbors (exclude self if src_ids == dst_ids)
            n_to_find = k + 1 if np.array_equal(src_ids, dst_ids) else k
            nns = np.argsort(dists)[:n_to_find]
            
            for d_l in nns:
                d_g = dst_ids[d_l]
                if s_g == d_g: continue # skip self-loops for geo edges
                
                src_e.append(local_map_src[s_g])
                dst_e.append(local_map_dst[d_g])
                if len(src_e) >= k * len(src_ids): break # limit to k per node
        
        return torch.tensor([src_e, dst_e], dtype=torch.long)

    # 1. Geo edges (intra-tier)
    hetero['macro','geo','macro'].edge_index = knn_edges(macro_idx, macro_idx, k)
    hetero['pico','geo','pico'].edge_index   = knn_edges(pico_idx, pico_idx, k)
    hetero['femto','geo','femto'].edge_index = knn_edges(femto_idx, femto_idx, k)
    
    # 2. Cross-tier edges (inter-tier influence)
    hetero['macro','cross','pico'].edge_index  = knn_edges(macro_idx, pico_idx, k)
    hetero['macro','cross','femto'].edge_index = knn_edges(macro_idx, femto_idx, k)
    
    return hetero, macro_idx, pico_idx, femto_idx

if __name__ == "__main__":
    # Test build
    data = np.random.rand(100, 100)
    hetero, m, p, f = build_hetero_graph(data)
    print(f"Nodes - Macro: {len(m)}, Pico: {len(p)}, Femto: {len(f)}")
    for edge_type in hetero.edge_types:
        print(f"Edge {edge_type} count: {hetero[edge_type].edge_index.shape[1]}")
