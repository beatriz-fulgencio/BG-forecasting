import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("results")
models = ["RNN", "LSTM", "GRU"]
patients = [540, 544, 552, 567, 584, 596]

mae_data = {model: [] for model in models}

for model in models:
    for patient in patients:
        fname = f"{model}_patient_{patient}_predictions.csv"
        fpath = results_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            # Adjust column names if needed
            y_true = df['true_values']
            y_pred = df['predictions']
            mae = np.mean(np.abs(y_true - y_pred))
            mae_data[model].append(mae)
        else:
            mae_data[model].append(np.nan)

# Plot
x = np.arange(len(patients))
bar_width = 0.25

plt.figure(figsize=(10,6))
for i, model in enumerate(models):
    plt.bar(x + i*bar_width, mae_data[model], width=bar_width, label=model)

plt.xlabel("Patient ID")
plt.ylabel("MAE (mg/dL)")
plt.title("Patient-specific model performance comparison (MAE in mg/dL)")
plt.xticks(x + bar_width, [str(p) for p in patients])
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "patient_performance_comparison.png", dpi=300)
plt.show()