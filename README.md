# 🧠 Soft K-Means Clustering using Fuzzy C-Means (FCM)

This Jupyter notebook implements **Soft K-Means Clustering** using the **Fuzzy C-Means (FCM)** algorithm from the `scikit-fuzzy` library.

Unlike traditional K-Means, where each data point belongs strictly to one cluster, FCM assigns **degrees of membership** — making it more flexible for overlapping or uncertain data.

---

## 📌 What’s Included

- ✅ Generation of synthetic 2D data for 3 clusters
- ✅ Application of Fuzzy C-Means clustering via `skfuzzy.cmeans`
- ✅ Calculation of **fuzzy cluster centers**
- ✅ Computation of **fuzzy radii** (based on weighted distances)
- ✅ Visualization of:
  - Clustered data points
  - Cluster centers
  - Soft cluster boundaries as transparent circles
- ✅ Evaluation using the **Fuzzy Partition Coefficient (FPC)**

---

## 📷 Example Output

Cluster visualization with fuzzy boundaries:

- Cluster centers marked with ✖️
- Circular regions show influence zones (computed fuzzy radii)

---

## 🛠 Dependencies

To run the notebook, you’ll need:

```bash
pip install numpy matplotlib scikit-fuzzy
