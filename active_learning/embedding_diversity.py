id="2lj93s"
import numpy as np
from sklearn.cluster import KMeans


def diversity_sampling(samples, n_clusters=10):

    if len(samples) == 0:
        return []

    embeddings = np.array(
        [s["embedding"] for s in samples]
    )

    kmeans = KMeans(
        n_clusters=min(n_clusters, len(samples)),
        random_state=42
    )

    kmeans.fit(embeddings)

    selected_samples = []

    for center in kmeans.cluster_centers_:

        distances = np.linalg.norm(
            embeddings - center,
            axis=1
        )

        idx = np.argmin(distances)

        selected_samples.append(samples[idx])

    return selected_samples