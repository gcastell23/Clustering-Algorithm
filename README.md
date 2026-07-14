# Clustering-Algorithm

# IGMC — Information-Geometric Multiscale Clustering

Welcome to the **IGMC** project showcase! We developed a noise-aware mathematical framework to cluster single-cell RNA-seq data and identify rare, transitioning cell populations in diabetic tissues without losing them to background noise.

## 🚀 The Core Pillars (Explained Simply)
1. **The Smart Ruler 📏:** Standard straight-line distance metrics ignore biological noise. Our model uses a customized mathematical ruler that bends around statistical noise, highlighting rare, low-expression cells.
2. **The Ink-Drop Test 💧:** Instead of choosing an arbitrary "resolution knob" to make clusters, we run random walks over our data. Real biological communities naturally trap the walker over multiple timescales.
3. **Islands vs. Highways 🛣️:** Our shape validation checks if cells belong to isolated groups (islands) or are dynamically mutating (highways).
4. **Calibrated Forecasts 🌤️:** We generate honest statistical "prediction sets" to identify exactly which cells have ambiguous or transitioning properties.

---

## 💻 How to Run the Code
You can run this algorithm directly on your machine or in **GitHub Codespaces** in seconds.

### 1. Install dependencies
```bash
pip install -r requirements.txt

Bash
python igmc_lite.py

## 🏃‍♂️ Fast Steps to Run and Publish This Live on GitHub

1. **Upload the Files:** Create a new repository on GitHub (e.g., `igmc-project`) and drag and drop these 4 files (`requirements.txt`, `igmc_lite.py`, `index.html`, `README.md`) directly into the repository.
2. **Open in the Cloud (To Run/Test):**
   * On your GitHub repository homepage, click the green **Code** button.
   * Switch to the **Codespaces** tab and click **Create codespace on main**.
   * When the cloud environment loads, paste these commands into the terminal window at the bottom to generate your visual results:
     ```bash
     pip install -r requirements.txt
     python igmc_lite.py
     ```
3. **Publish the Website (To Show Your Judges):**
   * Go back to your GitHub repository on GitHub.com.
   * Click **Settings** (the gear icon at the top right) ➡️ **Pages** (on the left menu).
   * Under "Branch", select **`main`** and click **Save**.
   * Refresh the page after 1 minute; you will see a link showing your live, interactive presentation dashboard!
