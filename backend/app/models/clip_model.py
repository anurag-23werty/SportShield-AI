import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_API_KEY")
)

def get_embedding(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = client.feature_extraction(
        image_bytes,
        model="openai/clip-vit-base-patch32"
    )

    return np.array(result).flatten()