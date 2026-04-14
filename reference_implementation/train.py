import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import load_telecom_italia
from graph_builder import build_hetero_graph
from model import TASTF

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './wireless dataset'
N_CELLS, SEQ_LEN, HORIZON = 100, 12, 3
BATCH_SIZE, EPOCHS, LR = 32, 50, 1e-3
PATIENCE = 10

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), scaler = \
        load_telecom_italia(DATA_DIR, N_CELLS, SEQ_LEN, HORIZON)
    
    X_tr  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_te  = torch.tensor(X_te,  dtype=torch.float32)
    y_te  = torch.tensor(y_te,  dtype=torch.float32)

    # 2. Build Graph (using training data mean activity)
    hetero, macro_idx, pico_idx, femto_idx = \
        build_hetero_graph(X_tr.numpy().reshape(-1, N_CELLS))
    hetero = hetero.to(DEVICE)
    
    print(f"Graph: {len(macro_idx)} Macro, {len(pico_idx)} Pico, {len(femto_idx)} Femto nodes")

    # 3. Initialize Model
    model = TASTF(N=N_CELLS, horizon=HORIZON,
                  macro_idx=macro_idx, pico_idx=pico_idx,
                  femto_idx=femto_idx).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    wait = 0
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0
        
        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
            
            opt.zero_grad()
            pred = model(xb, hetero)
            loss = loss_fn(pred, yb)
            loss.backward()
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE), hetero)
            val_loss = loss_fn(val_pred, y_val.to(DEVICE)).item()
        
        avg_train_loss = epoch_loss / (len(X_tr) / BATCH_SIZE)
        print(f'Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # Early Stopping & Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'tastf_best.pt')
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # 5. Final Evaluation
    model.load_state_dict(torch.load('tastf_best.pt'))
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te.to(DEVICE), hetero).cpu().numpy()
        te_true = y_te.numpy()
        
    mae  = np.mean(np.abs(te_pred - te_true))
    rmse = np.sqrt(np.mean((te_pred - te_true)**2))
    # MAPE with epsilon to avoid division by zero
    mape = np.mean(np.abs((te_pred - te_true) / (te_true + 1e-5))) * 100
    
    print("-" * 30)
    print(f"TEST RESULTS:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 30)
    
    # Save results for evaluate.py
    np.savez('results.npz', pred=te_pred, true=te_true, 
             macro=macro_idx, pico=pico_idx, femto=femto_idx)

if __name__ == "__main__":
    train()
