# Clustering-Algorithm

# IGMC — Information-Geometric Multiscale Clustering

Welcome to the **IGMC** project showcase! We developed a noise-aware mathematical framework to cluster single-cell RNA-seq data and identify rare, transitioning cell populations in diabetic tissues without losing them to background noise.

## 🚀 The Core Pillars (Explained Simply)
1. **The Smart Ruler 📏:** Standard straight-line distance metrics ignore biological noise. Our model uses a customized mathematical ruler that bends around statistical noise, highlighting rare, low-expression cells.
2. **The Ink-Drop Test 💧:** Instead of choosing an arbitrary "resolution knob" to make clusters, we run random walks over our data. Real biological communities naturally trap the walker over multiple timescales.
3. **Islands vs. Highways 🛣️:** Our shape validation checks if cells belong to isolated groups (islands) or are dynamically mutating (highways).
4. **Calibrated Forecasts 🌤️:** We generate honest statistical "prediction sets" to identify exactly which cells have ambiguous or transitioning properties.
