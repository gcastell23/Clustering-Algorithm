# IGMC — Information-Geometric Multiscale Clustering

Welcome to the **IGMC** project repository! Our work introduces a noise-aware mathematical framework to cluster single-cell RNA-seq data. By implementing an adaptive distance metric and topological shape tests, we isolate rare, transitioning cellular phenotypes (such as in patients with HNF1A-driven monogenic diabetes) that standard methods fail to detect.

---

## 🔬 Core Innovations

### Part 1: Biological & Computational Context
* **Foundational History:** Building on the droplet-based single-cell sequencing methods introduced by **Macosko et al. (2015)**, we study transcripts on a cellular level rather than using bulk averages.
* **The Integration Challenge:** Using advanced dataset harmonization concepts studied by multi-platform integrators like **FIRM (Ming et al., 2022)**, we process complex islet cells while preserving their unique variations.
* **Clinical Significance:** Our research focuses on how gene mutations push healthy cells into transitioning, intermediate states instead of simply eliminating them.

### Part 2: Algorithmic Advances
1. **The Smart Ruler:** We scale mathematical distances using a Negative Binomial distribution to highlight subtle low-expression cellular signals.
2. **The Ink-Drop Test:** Rather than using manual clustering parameters, continuous-time Markov stability random walks identify natural cellular groupings.
3. **Shape Testing:** We check if populations form isolated clusters (islands) or transitioning pathways (highways).
4. **Honest Predictions:** Conformal prediction sets tag transitioning cells with clear, calibrated ranges rather than overconfident labels.

---

## 💻 Running the Code

### 1. Install dependencies:
```bash
pip install -r requirements.txt
