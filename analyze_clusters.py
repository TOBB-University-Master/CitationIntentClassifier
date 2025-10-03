import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import os

os.makedirs("output", exist_ok=True)

# Embedding türünü buradan ayarlayabilirsin ("cls", "mean", "max", "attn")
embedding_type = "cls"      # Embedding type
timestamp = "202505200833"  # Dosya adındaki timestamp

# Veriyi yükle
embedding_path = f"output/{timestamp}_{embedding_type}_embedding.npy"
label_path = f"output/{timestamp}_labels.npy"

embeddings = np.load(embedding_path)
labels = np.load(label_path)

# Elbow
inertias = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi")
plt.savefig(f"output/{timestamp}_{embedding_type}_elbow.png", dpi=300)
plt.show()

# Silhouette
scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    score = silhouette_score(embeddings, kmeans.labels_)
    scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

best_k = k_values[np.argmax(scores)]
best_score = max(scores)

plt.figure(figsize=(8, 5))
bars = plt.bar(k_values, scores, color="lightgray")
bars[np.argmax(scores)].set_color("green")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette Skoru")
plt.title(f"Silhouette Skoru (En iyi: k={best_k}, skor={best_score:.4f})")
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"output/{timestamp}_{embedding_type}_silhouette.png", dpi=300)
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embeddings)
labels = np.array(labels)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
plt.colorbar(scatter)
plt.title("t-SNE (renklendirilmiş sınıflar)")
plt.savefig(f"output/{timestamp}_{embedding_type}_tsne.png", dpi=300)
plt.show()