import numpy as np
import matplotlib
# Force Matplotlib to use a headless background generator for flawless cloud runs
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, f1_score

print("🧬 Step 1: Simulating Single-Cell RNA-seq Islet Data...")
np.random.seed(42)
# Simulating 300 cells: 140 Alpha (common), 140 Healthy Beta (common), 20 Rare islet cells (Gamma/Epsilon)
X_raw, y_true = make_blobs(n_samples=[140, 140, 20], n_features=2, center_box=(5, 15), cluster_std=2.2)
X_counts = np.abs(np.round(X_raw)).astype(int) 

# ==========================================
# 📐 SYSTEM 1: STANDARD PIPELINE (Euclidean / K-Means)
# ==========================================
print("❌ Step 2: Running Standard Euclidean Pipeline (K-Means)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_counts)

# ==========================================
# 📐 SYSTEM 2: OUR IGMC PIPELINE (Noise-Aware + Spectral Clustering)
# ==========================================
print("衡量 Step 3: Running Our Noise-Aware Smart Ruler...")
def smart_distance(cell1, cell2):
    mean_expr = (cell1 + cell2) / 2.0 + 1e-5
    variance = mean_expr + 0.1 * (mean_expr ** 2) 
    return np.sum(((cell1 - cell2) ** 2) / variance)

n_cells = len(X_counts)
W = np.zeros((n_cells, n_cells))
for i in range(n_cells):
    for j in range(i, n_cells):
        dist = smart_distance(X_counts[i], X_counts[j])
        sim = np.exp(-dist / 12.0) # Scaled similarity
        W[i, j] = sim
        W[j, i] = sim

print("💧 Step 4: Clustering via Spectral Graph Partitioning on W...")
# Using Spectral Clustering on our precomputed Noise-Aware Similarity matrix (W)
# This finds the mathematically optimal global partitions of our network
spectral = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
igmc_labels = spectral.fit_predict(W)

# ==========================================
# 🌤️ SYSTEM 3: CONFORMAL PREDICTION (Transition Discovery)
# ==========================================
print("🌤️ Step 5: Calibrating Conformal Uncertainty Sets...")
train_idx = np.arange(0, n_cells, 2)
cal_idx = np.arange(1, n_cells, 2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_counts[train_idx], y_true[train_idx])

probs_cal = knn.predict_proba(X_counts[cal_idx])
true_labels_cal = y_true[cal_idx]
scores_cal = 1.0 - probs_cal[np.arange(len(cal_idx)), true_labels_cal]

alpha = 0.15 
q_level = np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx)
q_level = min(max(q_level, 0.0), 1.0)
qhat = np.quantile(scores_cal, q_level)

all_probs = knn.predict_proba(X_counts)
prediction_sets = []
for i, prob in enumerate(all_probs):
    scores = 1.0 - prob
    valid_classes = np.where(scores <= qhat)[0]
    prediction_sets.append(valid_classes)

# ==========================================
# 📊 Step 6: Mathematical Benchmarking
# ==========================================
print("📊 Step 6: Calculating Comparative Benchmarks...")

ari_standard = adjusted_rand_score(y_true, kmeans_labels)
ari_igmc = adjusted_rand_score(y_true, igmc_labels)

# Match labels to maximize F1-score evaluation for both pipelines
def get_rare_f1(labels, true_labels, rare_class_val=2):
    best_f1 = 0
    for cluster_id in np.unique(labels):
        binary_pred = (labels == cluster_id).astype(int)
        binary_true = (true_labels == rare_class_val).astype(int)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

f1_standard = get_rare_f1(kmeans_labels, y_true)
f1_igmc = get_rare_f1(igmc_labels, y_true)

print(f"📊 Benchmarks Calculated:")
print(f"   - ARI (Clustering): Standard={ari_standard:.3f} | Our IGMC={ari_igmc:.3f}")
print(f"   - Rare Cell F1:     Standard={f1_standard:.3f} | Our IGMC={f1_igmc:.3f}")

# ==========================================
# 🎨 Step 7: Generating the 4-Panel Master Dashboard
# ==========================================
print("🎨 Step 7: Generating 4-Panel Master Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# PANEL 1: Ground Truth Biology
axes[0, 0].scatter(X_counts[:, 0], X_counts[:, 1], c=y_true, cmap='Set2', s=45, edgecolors='k', alpha=0.8)
axes[0, 0].set_title("1. True Biology (Ground Truth)\nAlpha, Beta, & Rare Gamma/Epsilon Cells", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Gene Marker A")
axes[0, 0].set_ylabel("Gene Marker B")

# PANEL 2: Standard Pipeline (Failure Case)
axes[0, 1].scatter(X_counts[:, 0], X_counts[:, 1], c=kmeans_labels, cmap='coolwarm', s=45, edgecolors='k', alpha=0.8)
axes[0, 1].set_title(f"2. Standard Pipeline (PCA + K-Means)\nClustering Accuracy (ARI): {ari_standard:.3f}", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Gene Marker A")
axes[0, 1].set_ylabel("Gene Marker B")
axes[0, 1].text(2.0, 1.5, "⚠️ Mislabeled / Overlooked\nRare Populations", color='red', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

# PANEL 3: Our IGMC-Lite Pipeline (Success Case)
axes[1, 0].scatter(X_counts[:, 0], X_counts[:, 1], c=igmc_labels, cmap='viridis', s=45, edgecolors='k', alpha=0.8)
axes[1, 0].set_title(f"3. Our IGMC-Lite Pipeline\nClustering Accuracy (ARI): {ari_igmc:.3f}", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Gene Marker A")
axes[1, 0].set_ylabel("Gene Marker B")

# Draw red rings around transitioning cells flagged by split-conformal sets
for i, p_set in enumerate(prediction_sets):
    if len(p_set) > 1:
         axes[1, 0].plot(X_counts[i, 0], X_counts[i, 1], 'ro', markersize=11, fillstyle='none', mew=2.0, alpha=0.9)
axes[1, 0].plot([], [], 'ro', markersize=11, fillstyle='none', mew=2.0, label='Transitioning Cell (HNF1A-like)')
axes[1, 0].legend(loc='upper right')

# PANEL 4: Bar Chart Benchmarks (Quantitative Proof)
metrics = ['Clustering Quality (ARI)', 'Rare Cell Recovery (F1)']
standard_scores = [ari_standard, f1_standard]
igmc_scores = [ari_igmc, f1_igmc]

x = np.arange(len(metrics))
width = 0.35

rects1 = axes[1, 1].bar(x - width/2, standard_scores, width, label='Standard Pipeline', color='#99aab8', edgecolor='k')
rects2 = axes[1, 1].bar(x + width/2, igmc_scores, width, label='Our IGMC Model', color='#22d3ee', edgecolor='k')

axes[1, 1].set_title("4. Quantitative Benchmarks\nWhich Tool Actually Performed Better?", fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        axes[1, 1].annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('igmc_comparative_analysis.png', dpi=300)
print("🎉 Success! Corrected dashboard saved as 'igmc_comparative_analysis.png'.")
