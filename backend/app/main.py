from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import numpy as np
import mlflow
import time
import google.generativeai as genai
from dotenv import load_dotenv

from app.utils.hashing import generate_hash, hash_distance
from app.utils.tamper import tamper_score

# =========================
# ENV + GEMINI + MLFLOW
# =========================
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")
mlflow.set_experiment("SportShield")

# =========================
# APP
# =========================
app = FastAPI(title="SportShield AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

official_hash = None
official_path = None


# =========================
# HELPERS
# =========================
def histogram_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    hist1 = cv2.calcHist(
        [img1],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )

    hist2 = cv2.calcHist(
        [img2],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(score)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "SportShield backend running"}


@app.post("/register")
async def register_asset(file: UploadFile = File(...)):
    global official_hash, official_path

    path = f"{UPLOAD_DIR}/official.jpg"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    official_hash = generate_hash(path)
    official_path = path

    return {
        "status": "registered",
        "hash": official_hash
    }


@app.post("/scan")
async def scan_asset(file: UploadFile = File(...)):
    global official_hash, official_path
    start = time.time()

    try:
        if official_hash is None:
            return {"error": "Register official asset first"}

        path = f"{UPLOAD_DIR}/scan.jpg"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("saved scan")

        scan_hash = generate_hash(path)
        print("hash generated")

        # Hash similarity
        distance = hash_distance(official_hash, scan_hash)
        hash_score = max(0, 1 - (distance / 64))
        print("hash score:", hash_score)

        # Histogram similarity
        hist_score = histogram_similarity(
            official_path,
            path
        )
        print("hist score:", hist_score)

        # Tamper score
        tamper = tamper_score(
            official_path,
            path
        )
        print("tamper:", tamper)

        # Final score
        final_score = (
            (hash_score * 0.5) +
            (hist_score * 0.3) +
            ((1 - tamper) * 0.2)
        )

        # Risk label
        risk = "LOW"
        if final_score > 0.85:
            risk = "HIGH"
        elif final_score > 0.65:
            risk = "MEDIUM"

        # =========================
        # GEMINI ANALYSIS
        # =========================
        prompt = f"""
        Analyze this sports media authenticity report:

        Hash similarity: {hash_score}
        Histogram similarity: {hist_score}
        Tamper score: {tamper}
        Final score: {final_score}
        Risk level: {risk}

        Explain authenticity risk professionally in 2 short sentences.
        """

        response = gemini.generate_content(prompt)
        ai_explanation = response.text

        # =========================
        # MLFLOW LOGGING
        # =========================
        latency = time.time() - start

        with mlflow.start_run():
            mlflow.log_param("model_version", "hybrid-v1")
            mlflow.log_param("detector", "hash+histogram+tamper")

            mlflow.log_metric("hash_score", float(hash_score))
            mlflow.log_metric("histogram_score", float(hist_score))
            mlflow.log_metric("tamper_score", float(tamper))
            mlflow.log_metric("final_score", float(final_score))
            mlflow.log_metric("latency", float(latency))

        print("MLflow logged successfully")

        return {
            "hash_score": float(round(hash_score, 3)),
            "histogram_score": float(round(hist_score, 3)),
            "tamper_score": float(round(tamper, 3)),
            "final_score": float(round(final_score, 3)),
            "risk": str(risk),
            "match_found": bool(final_score > 0.65),
            "ai_explanation": ai_explanation
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            "error": str(e),
            "type": str(type(e))
        }