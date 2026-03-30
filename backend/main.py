import os
from pathlib import Path
from typing import List
 
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
 
# ── Model and vectorizer ──
model = None
vectorizer = None
 
def load_model():
    global model, vectorizer
    import pickle
    from xgboost import Booster

    model = Booster()
    model.load_model("model/sentiment_model.ubj")
    print("✅  Loaded XGBoost model")

    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅  Loaded vectorizer")
 
def run_sentiment(texts: List[str]) -> List[float]:
    import xgboost as xgb
    import numpy as np
    X = vectorizer.transform(texts)
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    preds = np.argmax(np.array(preds).reshape(-1, 3), axis=1)
    scores = []
    for p in preds:
        if p == 1:
            scores.append(1.0)
        elif p == 2:
            scores.append(-1.0)
        else:
            scores.append(0.0)
    return scores
 
 
# ── Topic extraction ──
TOPIC_KEYWORDS = {
    "Quality":          ["quality","build","material","made","sturdy","flimsy","durable"],
    "Delivery":         ["delivery","shipping","arrived","fast","slow","days","late"],
    "Value":            ["value","price","worth","money","cheap","expensive","affordable"],
    "Battery / Power":  ["battery","charge","power","life","drain"],
    "Packaging":        ["packaging","box","damaged","wrap","sealed"],
    "Customer service": ["customer service","support","help","return","refund"],
    "Ease of use":      ["easy","setup","install","instructions","simple","confusing"],
    "Size / Fit":       ["size","fit","small","large","bigger","smaller","tight"],
    "Appearance":       ["look","colour","color","design","style","appearance"],
}
 
def extract_topics(texts: List[str]) -> List[dict]:
    counts = {}
    for text in texts:
        lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in lower for k in keywords):
                counts[topic] = counts.get(topic, 0) + 1
    result = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"topic": t, "count": c} for t, c in result[:7]]
 
 
# ── API ──
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
    return {"status": "ok"}
 
 
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
        elif score == -1.0:
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
 
 
# ── Serve frontend ──
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
 
 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
 