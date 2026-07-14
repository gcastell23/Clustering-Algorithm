import numpy as np
import matplotlib
# Crucial: Force Matplotlib to use a 'headless' background generator 
# so it runs flawlessly in the cloud without needing a monitor!
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

print("🧬 Step 1: Simulating Single-Cell RNA-seq Islet Data...")
np.random.seed(42)
# Simulating 300 cells: 140 Alpha (common), 140 Healthy Beta (common), 20 Rare islet cells (Gamma/Epsilon)
X_raw, y_true = make_blobs(n_samples=[140, 140, 20], n_features=2, center_box=(0, 20), cluster_std=1.5)
X_counts = np.abs(np.round(X_raw)).astype(int) 

print("📏 Step 2: Running the Noise-Aware Smart Ruler (Information Geometry)...")
def smart_distance(cell1, cell2):
    # Biological counts follow a Negative Binomial noise property
    mean_expr = (cell1 + cell2) / 2.0 + 1e-5
    variance = mean_expr + 0.1 * (mean_expr ** 2) 
    # Scale differences by expected noise to pull out rare biological signals[cite: 1]
    return np.sum(((cell1 - cell2) ** 2) / variance)

# Generate our customized similarity network[cite: 1]
n_cells = len(X_counts)
W = np.zeros((n_cells, n_cells))
for i in range(n_cells):
    for j in range(i, n_cells):
        dist = smart_distance(X_counts[i], X_counts[j])
        sim = np.exp(-dist / 10.0) 
        W[i, j] = sim
        W[j, i] = sim

print("💧 Step 3: Simulating the Ink-Drop Test (Markov Random Walks)...")
row_sums = W.sum(axis=1)
P = W / row_sums[:, np.newaxis]

# Track where random walks naturally pool at timescale t=2[cite: 1]
P_t = np.linalg.matrix_power(P, 2)
predicted_clusters = np.argmax(P_t, axis=1)
unique_clusters, indexed_clusters = np.unique(predicted_clusters, return_inverse=True)

print("🌤️ Step 4: Calibrating Conformal Prediction Sets (Honest Forecasts)...")
train_idx = np.arange(0, n_cells, 2)
cal_idx = np.arange(1, n_cells, 2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_counts[train_idx], y_true[train_idx])

# Quantify uncertainty thresholds[cite: 1]
probs_cal = knn.predict_proba(X_counts[cal_idx])
true_labels_cal = y_true[cal_idx]
scores_cal = 1.0 - probs_cal[np.arange(len(cal_idx)), true_labels_cal]

alpha = 0.1 # Guarantee 90% accuracy coverage[cite: 1]
q_level = np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx)
q_level = min(max(q_level, 0.0), 1.0)
qhat = np.quantile(scores_cal, q_level)

# Pinpoint cells with transitioning phenotypes (Ambiguous cells)[cite: 1]
all_probs = knn.predict_proba(X_counts)
prediction_sets = []
ambiguous_count = 0

for i, prob in enumerate(all_probs):
    scores = 1.0 - prob
    valid_classes = np.where(scores <= qhat)[0]
    prediction_sets.append(valid_classes)
    if len(valid_classes) > 1:
        ambiguous_count += 1

print(f"📊 Dashboard Complete: Successfully flagged {ambiguous_count} transitioning cells.")

# Build & Save the High-Quality Visual Dashboard[cite: 1]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: True Biological Labels
scatter0 = axes[0].scatter(X_counts[:, 0], X_counts[:, 1], c=y_true, cmap='Set2', s=45, edgecolors='k', alpha=0.8)
axes[0].set_title("True Biology (Ground Truth Labels)\nCommon Cells vs. Rare Populations", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Gene Expression Marker A")
axes[0].set_ylabel("Gene Expression Marker B")

# Plot B: Our Algorithmic Discovery
scatter1 = axes[1].scatter(X_counts[:, 0], X_counts[:, 1], c=indexed_clusters, cmap='jet', s=45, edgecolors='k', alpha=0.8)
axes[1].set_title("IGMC-Lite Real-time Discovery\nHighlighted Transitioning/Ambiguous Cells", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Gene Expression Marker A")

# Circle the transition-state cells identified by Conformal Prediction[cite: 1]
flagged = False
for i, p_set in enumerate(prediction_sets):
    if len(p_set) > 1:
        axes[1].plot(X_counts[i, 0], X_counts[i, 1], 'ro', markersize=9, fillstyle='none', mew=1.5, alpha=0.7)
        flagged = True

axes[1].plot([], [], 'ro', markersize=9, fillstyle='none', mew=1.5, label='Mutated/Transitioning Cell (HNF1A-like)')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('igmc_results.png', dpi=300)
print("🎉 Success! Visual results saved as 'igmc_results.png'.")
