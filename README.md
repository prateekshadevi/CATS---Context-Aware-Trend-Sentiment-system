# CATS (Context-Aware Trend Sentiment system)
## Context-Aware Social Media Trend Prediction and Sentiment Analysis Using LLM-RAG

## 📺 Project Demonstration
[![CATS System Demo](https://img.youtube.com/vi/oyFKBfuYrgo/maxresdefault.jpg)](https://youtu.be/oyFKBfuYrgo)

*Click the image above to watch the full system walkthrough on YouTube, demonstrating our Tier 1 Deep RAG pipeline and Tier 2 Horizon Forecast.*

---

## Project Overview
CATS is an automated intelligence pipeline designed to solve the *temporal gap* in traditional AI by grounding Large Language Models in real-time data. The system identifies emerging search trends and synthesizes a 360-degree context—spanning factual reporting, public emotion, and predictive momentum—to provide high-fidelity summaries and virality forecasts.

## Core Architecture

### 1. Dynamic Seeding & Tiered Ingestion
The system is initialized using **Google Trends** "Breakout" data (surges > 5,000%). 
* **Tier 1 (Deep Context):** The Top 10 trends undergo full-text retrieval from NewsAPI, Wikipedia, Reddit, and Twitter (Bluesky) for RAG summarization and sentiment analysis.
* **Tier 2 (Momentum):** The Top 100 trends are processed for engagement metadata (likes, reposts, timestamps) to feed the predictive model.

### 2. The "Strict" RAG Pipeline
To eliminate hallucination, CATS utilizes **Qwen 2.5 7B (Instruct-Quantized)** deployed locally on the Northeastern Discovery Cluster.
* **Hybrid Retrieval:** Combines **BGE-M3** vector similarity with a *Keyword Lock* to ensure strict topical relevance.
* **Scientific Auditor:** A custom secondary LLM layer that cross-references summaries against source chunks to produce *Faithfulness* and *Relevancy* scores.

### 3. Hybrid Sentiment Analysis
A dual-model pipeline that balances speed and reasoning:
* **Primary:** `twitter-roberta-base-sentiment-latest` for high-speed inference.
* **Escalation:** Ambiguous or sarcastic content (confidence < 0.65) is routed to Qwen 7B for deep linguistic reasoning.
* **Weighting:** Final labels are weighted by **Engagement (Buzz)** and **Exponential Recency Decay**.

### 4. XGBoost Trend Prediction
A machine learning engine that forecasts trend trajectory using:
* **Features:** Logarithmic velocity, maturity (duration), breakdown counts, and matchup detection.
* **Output:** Identifies **Dark Horse** candidates, topics outside the Top 10 with high acceleration, and calculates survival probabilities for the next 24 hours.

## Repository Structure
* `CATS.ipynb`: The primary end-to-end integration notebook containing the collection, RAG, sentiment, and prediction logic.
* `dashboard.py`: Streamlit-based interface for visualizing real-time intelligence feeds and trend momentum.
* `/data/raw/`: Directory containing `trends_24.csv` and `trends_48.csv` used for model training and query seeding.
* `chroma_db_top_10/`: Persistent vector store containing semantically filtered text chunks for the Tier 1 trends.

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/prateekshadevi/CATS---Context-Aware-Trend-Sentiment-system.git](https://github.com/prateekshadevi/CATS---Context-Aware-Trend-Sentiment-system.git)
