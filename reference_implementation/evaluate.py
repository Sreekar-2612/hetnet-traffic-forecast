import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_tier_predictions(file_path='results.npz'):
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Results file {file_path} not found. Run train.py first.")
        return

    te_pred = data['pred']
    te_true = data['true']
    macro_idx = data['macro']
    pico_idx = data['pico']
    femto_idx = data['femto']
    
    # Take a sample from the test set for visualization
    # Shape of te_pred/te_true: (Samples, Horizon, Nodes)
    # We'll plot the first 100 timesteps for the first BS of each tier
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Indices to plot (first BS in each list)
    plot_indices = [macro_idx[0], pico_idx[0], femto_idx[0]]
    labels = ['Macro BS', 'Pico BS', 'Femto BS']
    colors = ['#FF4B4B', '#1C83E1', '#00D67D'] # Modern vibrant colors
    
    # Horizon index to plot (e.g., 0 for the next timestep prediction)
    h = 0 
    
    for i, (idx, label, color) in enumerate(zip(plot_indices, labels, colors)):
        ax = axes[i]
        
        # Ground truth
        truth = te_true[:100, h, idx]
        # Prediction
        pred = te_pred[:100, h, idx]
        
        ax.plot(truth, label='Ground Truth', color='black', alpha=0.5, linewidth=2)
        ax.plot(pred, '--', label='TASTF Forecast', color=color, linewidth=2)
        
        ax.set_title(f'{label} Activity (Cell {idx})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (10-min Intervals)', fontsize=12)
        ax.set_ylabel('Normalized Activity', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
    plt.tight_layout()
    plt.savefig('tier_predictions.png', dpi=300)
    print("Tier predictions visualization saved as 'tier_predictions.png'")

if __name__ == "__main__":
    plot_tier_predictions()
