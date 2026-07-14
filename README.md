🧬 IGMC-Lite: Uncertainty-Aware Single-Cell Clustering
An advanced, high-performance computational pipeline for single-cell RNA sequencing (scRNA-seq) analysis. IGMC-Lite implements a robust, density-balanced Shared Nearest Neighbor (SNN) graph topology paired with Spectral Clustering, calibrated using Split-Conformal Prediction to identify cellular transition states with high statistical selectivity.

🚀 Key Features & Evolutionary Milestones
Our pipeline was engineered to address major pitfalls common in traditional single-cell clustering methods (e.g., standard PCA + K-Means):

Feature	The Old/Standard Way	The IGMC-Lite Way
Graph Topology	Euclidean distances on a flat coordinate grid (leads to spatial metric distortion).	Shared Nearest Neighbor (SNN) graph matching local neighborhood overlap.
Clustering Engine	K-Means (assumes clusters are spherical, clean, and equal-sized).	Normalized Spectral Manifold Partitioning to capture complex, non-linear biological shapes.
Uncertainty Tracking	Flat distance-to-centroid thresholds or over-sensitive conformal testing.	Calibrated Split-Conformal Decision Sets (α=0.15) highlighting true, selective boundary transitions.
Pipeline Reliability	Crashes on synthetic control datasets due to rigid biological gene filtering rules.	Dynamic Preprocessing Routing to bypass filtering on synthetic sets while fully processing real scRNA-seq.
🏛️ The Four Pillars of the Architecture
1. Density-Balanced Graph Topology (SNN)
Instead of relying on absolute Euclidean distances (which suffer from the "curse of dimensionality" in high dimensions), we calculate similarity based on local graph intersections. The similarity between cell i and cell j is computed as:

W 
ij
​
 = 
∣N(i)∪N(j)∣
∣N(i)∩N(j)∣
​
 
Where N(x) is the set of nearest neighbors for cell x. This Shared Nearest Neighbor (SNN) construction naturally dampens background noise and scales to different cell-type densities.

2. Normalized Spectral Manifold
We feed the SNN similarity matrix W into a precomputed Spectral Clustering algorithm. This solves the generalized eigenvector problem on the graph Laplacian, allowing the pipeline to easily trace complex, non-spherical manifolds that linear methods completely miss.

3. Calibrated Split-Conformal Prediction
To pinpoint "transitioning" or ambiguous cells (e.g., stem cells in differentiation pathways), we split our dataset and train a localized classifier to compute non-conformity scores:

s 
i
​
 =1.0− 
P
^
 (y 
i
​
 ∣X 
i
​
 )
At a controlled error tolerance of α=0.15 (85% confidence level), we define the prediction set. A cell is cleanly flagged as transitioning only if its conformal prediction set contains multiple classes (size≥2), successfully isolating true biological boundaries.

4. Multi-Dataset Quantitative Benchmarking
To prove generalization, the pipeline automatically evaluates and saves a 4-panel master dashboard for three diverse datasets in parallel:

Synthetic Blobs: Pure mathematical clustering control (1,000 cells).

10x PBMC 3k: Real-world peripheral blood mononuclear cells benchmark (approx. 2,700 cells).

PBMC 68k Reduced: Large-scale real-world scRNA-seq challenge (highly overlapping, complex subpopulation structures).

📁 Repository Structure
Code snippet
├── igmc_lite.py         # Main pipeline engine (SNN, Spectral, Conformal)
├── index.html           # Interactive, web-based presentation dashboard
├── README.md            # Project documentation (This file)
├── igmc_blobs.png       # Output Dashboard: Synthetic control
├── igmc_pbmc3k.png      # Output Dashboard: Real-world standard
└── igmc_pbmc68k.png     # Output Dashboard: Real-world large scale
