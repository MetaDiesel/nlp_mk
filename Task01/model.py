import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Reviews.csv")  # Ensure the file is in your working directory
df = df[['Text', 'Score']]  # Keep only relevant columns
df = df.dropna()  # Remove missing values

# Convert Score into binary sentiment (1: Positive, 0: Negative)
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

# Select a subset for faster processing
df = df.head(5000)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# TF-IDF Training and Logisiting Reg.Model
# Create a text processing and classification pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),  # Convert text to TF-IDF features
    ("classifier", LogisticRegression(max_iter=1000))  # Train a Logistic Regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model to disk using Pickle
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved as sentiment_model.pkl")

# Load the saved model
with open("sentiment_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Test with a sample review
sample_review = ["This product is amazing! I love it."]
prediction = loaded_model.predict(sample_review)
print(f"Sentiment Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")

