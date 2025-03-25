# Non-Linear Clustering Analysis on Synthetic and mHealth eDOL Datasets

This repository contains Jupyter notebooks for analyzing **non-linear clustering methods**, focusing on their performance in overlapping clusters and result reproducibility. The experiments are conducted on:

- **Synthetic Toy Dataset** (`toys-dataset.ipynb`)
- **Real-World mHealth eDOL Dataset** (`eDOL_dataset.ipynb`)

## Project Overview

Clustering is a fundamental task in data analysis and machine learning, used to uncover hidden structures within data. **Non-linear clustering methods**—such as **Kernel K-Means, Spectral Clustering, Density-Based Clustering (DBSCAN), and Deep Embedded Clustering**—are particularly effective in capturing complex patterns. However, their performance on **overlapping clusters** and **reproducibility** has been underexplored.

This research systematically evaluates multiple **non-linear clustering techniques** on:

- **Synthetic datasets** (toys-dataset.ipynb) to test algorithm behavior in controlled scenarios.
- **The real-world mHealth eDOL dataset** (eDOL_dataset.ipynb) to analyze cluster validity in real applications.

Ensure you have **Python 3.x** and the required libraries installed before running the notebooks:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
