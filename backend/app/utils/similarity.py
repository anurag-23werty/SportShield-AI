import numpy as np

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return 0

    return float(dot / norm)