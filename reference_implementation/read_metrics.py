import numpy as np

data = np.load('results.npz')
te_pred = data['pred']
te_true = data['true']

mae  = np.mean(np.abs(te_pred - te_true))
rmse = np.sqrt(np.mean((te_pred - te_true)**2))
mape = np.mean(np.abs((te_pred - te_true) / (te_true + 1e-5))) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
