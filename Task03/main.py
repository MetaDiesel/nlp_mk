from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load trained sentiment model
model = joblib.load("sentiment_model.pkl")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# API endpoint for sentiment analysis
@app.get("/predict")
def predict_sentiment(text: str):
    review_text = [text]  # Model expects a list
    sentiment_score = model.predict_proba(review_text)[:, 1][0]  # Probability of positive sentiment
    sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"
    return {"sentiment_score": sentiment_score, "sentiment": sentiment_label}
