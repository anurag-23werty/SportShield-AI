from PIL import Image
import imagehash

def generate_hash(image_path):
    img = Image.open(image_path)
    return str(imagehash.phash(img))

def hash_distance(hash1, hash2):
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)