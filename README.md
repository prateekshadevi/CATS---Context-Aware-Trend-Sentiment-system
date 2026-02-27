# CATS (Context-Aware Trend Sentiment system)
## Context-Aware Social Media Trend Prediction and Sentiment Analysis Using LLM-RAG

## Project Overview
This capstone project develops a Retrieval-Augmented Generation (RAG) system designed to identify emerging search trends and synthesize a 360-degree context from news, social media, and technical discourse. The system aims to provide high-fidelity summaries and sentiment analysis by grounding Large Language Models (LLMs) in real-time, multi-platform data.

## Iteration 3 Focus: Dataset Architecture & Selection
This iteration establishes the foundational dataset schema and the multi-source ingestion pipeline. 

### 1. Trend Discovery (The Seed)
Unlike static datasets, this project utilizes a dynamic seeding process:
* **Source:** Google Trends (Real-Time Search Data).
* **Methodology:** Manual export of the **"Past 24 Hours"** breakout trends in CSV format.
* **Rationale:** This high-granularity window captures "Breakout" topics (surges > 5,000%) as they unfold, ensuring the downstream data collection is synchronized with peak public interest.

### 2. Multi-Platform Contextual Sources
For every identified trend, the system retrieves data from:
* **NewsAPI:** Provides authoritative journalistic context and factual grounding.
* **Reddit API:** Captures broad public sentiment and emotional discourse.
* **Hacker News (Algolia API):** Acts as a technical sentiment proxy to filter hype from professional adoption.
* **Wikipedia API:** Provides baseline entity definitions to prevent LLM hallucinations.

## Repository Structure
* `/data/raw/`: Contains the Google Trends CSV exports used for query seeding.
* `data_dictionary.md`: Detailed definitions of the dataset schema and metadata.
* `requirements.txt`: Python dependencies required for the collection pipeline.
* `collector.py`: The core API integration logic for multi-source retrieval.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/prateekshadevi/CATS---Context-Aware-Trend-Sentiment-system.git](https://github.com/prateekshadevi/CATS---Context-Aware-Trend-Sentiment-system.git)
