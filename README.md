# dbscan 🦀🐍

 ## A Python clustering library written in Rust.

`dbscan` is a Python library designed for efficient density-based clustering. By leveraging a Rust-powered engine, it provides a significant speed boost over pure Python implementations for the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm, while remaining easy to use within the `NumPy` ecosystem.

 ## Academic Context

This project was developed as part of the **Second Year Integration Project (Projet Intégrateur)** of the **CMI Informatique (Cursus Master en Ingénierie)** at the **University of Reims Champagne-Ardenne (URCA)**, under the supervision of **Mr. Jean-Charles Boisson**.

**Features & Technologies**

- Rust Core: Implements the clustering logic using native data structures and optimized stack-based expansion for performance.

- Python Bindings (PyO3): Seamless integration with Python using PyO3 and rust-numpy, allowing direct handling of NumPy arrays.

- Advanced Prediction: Beyond standard clustering, the model includes predict and predict_with_epsilon methods to classify new, unseen data points using a k-Nearest Neighbors approach.

- Distance Metrics: Includes a dedicated distance_metrics module with optimized kernels for MAE (Manhattan) and RMSD (Euclidean) distances across 1D, 2D, 3D, and nD dimensions.

- Outlier Detection: Automatically identifies noise in the dataset, labeled with the usize::MAX constant.

 ## Installation

Prerequisites

- Rust & Cargo

- Python 3.7+

- maturin (to build the bridge between Rust and Python)
  
**Compilation**

To install the library into your current Python environment:

```bash
pip install maturin
maturin develop --release
```

 ## Usage in Python

```python
import numpy as np
from dbscan import DbscanModel

# Prepare your data
data = np.array([
    [0.1, 0.2], [0.2, 0.1], # Cluster 0
    [1.0, 1.1], [1.1, 1.0], # Cluster 1
    [10.0, 10.0]            # Outlier (Noise)
], dtype=np.float64)

# Initialize and train
model = DbscanModel()
model.train(data, epsilon=0.5, min_neighbor=2)

# Predict clusters for new points
# [0.15, 0.15] is near Cluster 0
# [9.0, 9.0] is nearest to Cluster 1 (since the outlier at [10,10] is ignored)
new_points = np.array([[0.15, 0.15], [9.0, 9.0]], dtype=np.float64)
labels = model.predict(new_points, n_neighbor=1)

print(f"Predicted labels: {labels}")
# Output: Predicted labels: [0 1]
```

 ## License

Distributed under the MIT License.