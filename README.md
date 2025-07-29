# Fake_Review_Detector

A Flask web application that uses a logistic regression model on TF–IDF features to classify product reviews as genuine or fake. This project is a work in progress: fine-tuning BERT, adding sentiment analysis, and building a real-time prediction pipeline.

## Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Performance](#model-performance)  
- [How to Improve](#how-to-improve)  
- [Next Steps (WIP)](#next-steps-wip)  

## Features

- Trains a logistic regression classifier on a labeled review dataset  
- Uses TF–IDF vectorization for text features  
- Provides a simple Flask web interface for live review classification  

## Project Structure
├── README.md
├── test_data.csv # CSV with columns: category, rating, label_raw, review text
├── fake_review-1.py # Script to train & evaluate model
├── app.py # Flask application
└── templates/
└── index.html # Web UI template


## Installation

1. Clone the repository
2. Create a virtual environment and install dependencies
3. Ensure `test_data.csv` is in the project root.

## Usage

1. Train the model (optional, since `app.py` trains at startup)
   
2. Run the Flask app: Pyhton app.py
3. Open your browser at `http://127.0.0.1:5000/` and enter a review to see “Fake” or “Genuine” predictions.

## Model Performance

| Metric    | Score     |
|-----------|-----------|
| Accuracy  | 90.4%     |
| Precision | 0.91      |
| Recall    | 0.89      |
| F1-score  | 0.90      |

## Working on / Coming updates:

1. **Fine-tuning BERT**  
- Use Hugging Face’s Transformers to fine-tune a pre-trained BERT model on your labeled reviews.  
2. **Add Sentiment Analysis**  
- Compute sentiment scores (e.g., via Vader or TextBlob) and include as additional features in the model.  
3. **Incorporate CNN for Text**  
- Use a 1D convolutional network on word embeddings to capture local n-gram patterns.  
4. **Build Real-Time System**  
- Containerize with Docker  
- Deploy with FastAPI and RSVP streaming (Kafka) for live review ingestion  
- Add Redis caching for low-latency inference  

## Next Steps (WIP)

- Fine-tune BERT for improved contextual understanding  
- Integrate sentiment-analysis pipelines  
- Implement real-time streaming architecture with Docker, Kafka, and FastAPI  

 







