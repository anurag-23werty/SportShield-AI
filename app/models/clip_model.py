from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

model = SentenceTransformer("clip-ViT-B-32")

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    embedding = model.encode(image)
    return np.array(embedding)