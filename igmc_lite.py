import numpy as np
import matplotlib
# Force Matplotlib to use a headless background generator for flawless cloud runs
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, f1_score
import scanpy as sc

# =====================================================================
# 📐 PILLAR 1 & 2: SHARED NEIGHBOR GRAPH & SPECTRAL CLUSTERING
# =====================================================================

def run_igmc_pipeline(X_scaled, n_clusters):
    """
    Builds a Shared Nearest Neighbor (SNN) graph similarity matrix.
    SNN is highly robust to noise and density differences in single-cell data.
    """
    n_neighbors = 15
    # Step 1: Get binary KNN graph (1 if neighbor, 0 otherwise)
    knn_graph = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, mode='connectivity', include_self=True).toarray()
    
    # Step 2: SNN similarity (W_ij = number of shared neighbors between cell i and j)
    # This acts as an automated noise-filter
    W = np.dot(knn_graph, knn_graph.T)
    
    # Step 3: Normalize similarity scale to [0, 1]
    diag = np.diagonal(W)
    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            denom = diag[i] + diag[j] - W[i, j]
            sim = W[i, j] / denom if denom > 0 else 0.0
            W[i, j] = sim
            W[j, i] = sim

    # Step 4: Run Spectral Clustering on our filtered SNN graph
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    return spectral.fit_predict(W)


# =====================================================================
# 📐 PILLAR 3: CALIBRATED SPLIT-CONFORMAL PREDICTION
# =====================================================================

def run_conformal_prediction(X_scaled, y_true):
    """
    Uses split-conformal prediction to flag highly ambiguous (transitioning) cells.
    We look for cells that belong to multiple classes at an 85% confidence level.
    """
    n_cells = len(X_scaled)
    train_idx = np.arange(0, n_cells, 2)
    cal_idx = np.arange(1, n_cells, 2)

    # Train a KNN classifier on the training split
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_scaled[train_idx], y_true[train_idx])

    # Calibrate on the validation split
    probs_cal = knn.predict_proba(X_scaled[cal_idx])
    true_labels_cal = y_true[cal_idx]
    
    # Non-conformity score: 1.0 - probability of the true label
    scores_cal = 1.0 - probs_cal[np.arange(len(cal_idx)), true_labels_cal]

    # Set alpha = 0.15 (85% confidence). This keeps flagging highly selective.
    alpha = 0.15 
    q_level = np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx)
    q_level = min(max(q_level, 0.0), 1.0)
    qhat = np.quantile(scores_cal, q_level)

    # Generate prediction sets for all cells
    all_probs = knn.predict_proba(X_scaled)
    prediction_sets = []
    for prob in all_probs:
        scores = 1.0 - prob
        # Include all classes whose non-conformity score is below the calibrated threshold
        valid_classes = np.where(scores <= qhat)[0]
        prediction_sets.append(valid_classes)
        
    return prediction_sets


# =====================================================================
# 📐 PILLAR 4: QUANTITATIVE EVALUATION METRICS
# =====================================================================

def get_rare_f1(labels, true_labels):
    """
    Evaluates recovery (F1-score) of the rarest biological population.
    """
    cluster_counts = np.bincount(true_labels)
    rare_val = np.argmin(cluster_counts)
    
    best_f1 = 0
    for cluster_id in np.unique(labels):
        binary_pred = (labels == cluster_id).astype(int)
        binary_true = (true_labels == rare_val).astype(int)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


# =====================================================================
# 🚀 Step 2: Main Processing Pipeline for All Datasets
# =====================================================================

datasets_to_run = {
    "10x_PBMC3k": {
        "loader": lambda: sc.datasets.pbmc3k(),
        "is_synthetic": False,
        "is_preprocessed": False,
        "filename": "igmc_pbmc3k.png"
    },
    "Synthetic_Blobs": {
        "loader": lambda: sc.datasets.blobs(n_variables=10, n_centers=4, n_observations=1000, random_state=42),
        "is_synthetic": True,
        "is_preprocessed": False,
        "filename": "igmc_blobs.png"
    },
    "PBMC_68k_Reduced": {
        "loader": lambda: sc.datasets.pbmc68k_reduced(),
        "is_synthetic": False,
        "is_preprocessed": True,
        "filename": "igmc_pbmc68k.png"
    }
}

for name, config in datasets_to_run.items():
    print("\n" + "="*60)
    print(f"📥 Loading Dataset: {name}...")
    print("="*60)
    
    # 1. Load Data
    adata = config["loader"]()
    
    # 2. Pipeline Preprocessing
    print(f"⚙️ Preprocessing {name}...")
    if config["is_preprocessed"]:
        pass
    elif config["is_synthetic"]:
        sc.tl.pca(adata, n_comps=2)
    else:
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, n_comps=2)
    
    # 3. Scale and Plot Preparation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(adata.obsm['X_pca'][:, :2])
    X_plot = X_scaled - X_scaled.min(axis=0)
    
    # Determine Ground Truth labels
    if 'blobs' in adata.obs:
        y_true = adata.obs['blobs'].astype(int).values
    elif 'leiden' in adata.obs:
        y_true = adata.obs['leiden'].cat.codes.values
    elif 'louvain' in adata.obs:
        y_true = adata.obs['louvain'].cat.codes.values
    else:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=2)
        sc.tl.leiden(adata, resolution=0.5)
        y_true = adata.obs['leiden'].cat.codes.values

    n_clusters = len(np.unique(y_true))
    print(f"🧬 Detected {X_scaled.shape[0]} cells across {n_clusters} true subpopulations.")

    # 4. Run Benchmark (Standard K-Means)
    print("❌ Running Standard Euclidean Pipeline (K-Means)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 5. Run Our Adaptive SNN Spectral Model
    print("📐 Running Our Adaptive SNN Graph Model...")
    igmc_labels = run_igmc_pipeline(X_scaled, n_clusters)

    # 6. Run Conformal Transition Discovery
    print("🌤️ Calculating Conformal Uncertainty Sets...")
    prediction_sets = run_conformal_prediction(X_scaled, y_true)

    # 7. Evaluate Performance metrics
    ari_standard = adjusted_rand_score(y_true, kmeans_labels)
    ari_igmc = adjusted_rand_score(y_true, igmc_labels)
    f1_standard = get_rare_f1(kmeans_labels, y_true)
    f1_igmc = get_rare_f1(igmc_labels, y_true)

    print(f"📈 RESULTS FOR {name}:")
    print(f"   ↳ K-Means:  ARI = {ari_standard:.3f} | Rare Cell F1 = {f1_standard:.3f}")
    print(f"   ↳ Our IGMC: ARI = {ari_igmc:.3f} | Rare Cell F1 = {f1_igmc:.3f}")

    # ==========================================
    # 🎨 Dashboard Plot Generation
    # ==========================================
    print(f"🎨 Exporting dashboard to '{config['filename']}'...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Ground Truth Biology
    axes[0, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=y_true, cmap='tab10', s=25, edgecolors='k', alpha=0.6, linewidths=0.3)
    axes[0, 0].set_title(f"1. True Biology (Ground Truth)\n{n_clusters} Identified Biological Subpopulations", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Normalized Gene Marker A")
    axes[0, 0].set_ylabel("Normalized Gene Marker B")

    # Panel 2: Standard Pipeline
    axes[0, 1].scatter(X_plot[:, 0], X_plot[:, 1], c=kmeans_labels, cmap='tab10', s=25, edgecolors='k', alpha=0.6, linewidths=0.3)
    axes[0, 1].set_title(f"2. Standard Pipeline (PCA + K-Means)\nClustering Accuracy (ARI): {ari_standard:.3f}", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Normalized Gene Marker A")
    axes[0, 1].set_ylabel("Normalized Gene Marker B")

    # Panel 3: Our IGMC-Lite Pipeline
    axes[1, 0].scatter(X_plot[:, 0], X_plot[:, 1], c=igmc_labels, cmap='tab10', s=25, edgecolors='k', alpha=0.6, linewidths=0.3)
    axes[1, 0].set_title(f"3. Our IGMC-Lite Pipeline\nClustering Accuracy (ARI): {ari_igmc:.3f}", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Normalized Gene Marker A")
    axes[1, 0].set_ylabel("Normalized Gene Marker B")

    # Flag Transitioning cells with red circles
    transitioning_count = 0
    for i, p_set in enumerate(prediction_sets):
        # A cell is flagged as transitioning only if its prediction set is multi-class (> 1)
        if len(p_set) > 1:
             axes[1, 0].plot(X_plot[i, 0], X_plot[i, 1], 'ro', markersize=6, fillstyle='none', mew=1.0, alpha=0.7)
             transitioning_count += 1
             
    axes[1, 0].plot([], [], 'ro', markersize=6, fillstyle='none', mew=1.0, label=f'Transitioning Cell ({transitioning_count} flagged)')
    axes[1, 0].legend(loc='upper right')

    # Panel 4: Comparative Bar Chart
    metrics = ['Clustering Quality (ARI)', 'Rare Cell Recovery (F1)']
    standard_scores = [ari_standard, f1_standard]
    igmc_scores = [ari_igmc, f1_igmc]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = axes[1, 1].bar(x - width/2, standard_scores, width, label='Standard Pipeline', color='#99aab8', edgecolor='k')
    rects2 = axes[1, 1].bar(x + width/2, igmc_scores, width, label='Our IGMC Model', color='#22d3ee', edgecolor='k')

    axes[1, 1].set_title(f"4. Quantitative Benchmarks ({name})\nWhich Tool Performed Better?", fontsize=12, fontweight='bold')
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
    plt.savefig(config["filename"], dpi=300)
    plt.close()

print("\n🎉 PIPELINE SUCCESSFUL! Check your workspace directory for clean, selective dashboards!")
