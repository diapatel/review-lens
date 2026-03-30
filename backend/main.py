"""
ReviewLens — FastAPI backend
Loads your saved sentiment model and serves the dashboard.

Supports:
  - scikit-learn pipelines / vectorizer+classifier combos (.pkl)
  - HuggingFace transformers models (.pt / saved directory)
  - Any model that exposes a predict() or __call__() method

Edit the `load_model` and `run_sentiment` functions below to match
your specific model format.
"""

import os
import re
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# 1.  MODEL LOADING
#     Edit this section to match your model file.
# ─────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model/sentiment_model.pkl")

model = None          # your model object
vectorizer = None     # only needed if you have a separate vectorizer .pkl


def load_model():
    global model, vectorizer
    import xgboost as xgb
    import pickle

    model = xgb.Booster()
    model.load_model("model/sentiment_model.pkl")
    print("✅  Loaded XGBoost model")

    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅  Loaded vectorizer")

def run_sentiment(texts: List[str]) -> List[float]:
    import numpy as np
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return [float(p) for p in np.array(preds).flatten()]

def keyword_score(text: str) -> float:
    """Simple keyword heuristic used when no model is loaded."""
    lower = text.lower()
    pos_words = ["love","amazing","great","excellent","perfect","best",
                 "fantastic","wonderful","brilliant","good","fast","easy",
                 "recommend","pleased","happy","awesome","outstanding"]
    neg_words = ["terrible","broke","bad","poor","disappoint","awful","worst",
                 "return","damaged","cheap","wrong","confusing","horrible",
                 "broken","useless","waste","slow","rude","never"]
    score = 0.0
    for w in pos_words:
        if w in lower: score += 0.18
    for w in neg_words:
        if w in lower: score -= 0.20
    stars = len(re.findall(r"⭐", text))
    if stars:
        score += (stars - 3) * 0.15
    return round(max(-1.0, min(1.0, score)), 3)


# ─────────────────────────────────────────────
# 2.  TOPIC EXTRACTION
#     Add or edit topics to match your product category.
# ─────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "Quality":          ["quality","build","material","made","construction","sturdy","flimsy","durable"],
    "Delivery":         ["delivery","shipping","arrived","fast","slow","days","dispatch","courier","late"],
    "Value":            ["value","price","worth","money","cheap","expensive","cost","affordable","overpriced"],
    "Battery / Power":  ["battery","charge","power","life","drain","charging","plug"],
    "Packaging":        ["packaging","box","damaged","wrap","sealed","opening"],
    "Customer service": ["customer service","support","help","return","refund","response","contact"],
    "Ease of use":      ["easy","setup","install","instructions","simple","complicated","confusing","intuitive"],
    "Size / Fit":       ["size","fit","small","large","dimension","bigger","smaller","tight","loose"],
    "Appearance":       ["look","colour","color","design","style","photo","picture","appearance","finish"],
}


def extract_topics(texts: List[str]) -> List[dict]:
    counts: dict[str, int] = {}
    for text in texts:
        lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in lower for k in keywords):
                counts[topic] = counts.get(topic, 0) + 1
    result = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"topic": t, "count": c} for t, c in result[:7]]


# ─────────────────────────────────────────────
# 3.  API
# ─────────────────────────────────────────────

app = FastAPI(title="ReviewLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    reviews: List[str]


class ReviewResult(BaseModel):
    text: str
    score: float
    label: str


class AnalyseResponse(BaseModel):
    results: List[ReviewResult]
    topics: List[dict]
    summary: dict


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    if len(req.reviews) > 5000:
        raise HTTPException(status_code=400, detail="Max 5000 reviews per request.")
    scores = run_sentiment(req.reviews)
    results = []
    for text, score in zip(req.reviews, scores):
        if score == 1.0:
            label = "positive"
        elif score == 2.0:
            label = "negative"
        else:
            label = "neutral"
        results.append(ReviewResult(text=text, score=score, label=label))
    pos = sum(1 for r in results if r.label == "positive")
    neu = sum(1 for r in results if r.label == "neutral")
    neg = sum(1 for r in results if r.label == "negative")
    avg = round(sum(r.score for r in results) / len(results), 3)
    topics = extract_topics(req.reviews)
    return AnalyseResponse(
        results=results,
        topics=topics,
        summary={
            "total": len(results),
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "average_score": avg,
        },
    )


# Serve the frontend (static files) from ../frontend/
FRONTEND_DIR = Path(r"C:\Users\Rakesh\OneDrive\Desktop\reviewlens\frontend")
print(f"Frontend dir: {FRONTEND_DIR}, exists: {FRONTEND_DIR.exists()}")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"message": "ReviewLens API running. Put your frontend in ../frontend/"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
