# Data Dictionary: Trend-Context Dataset

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `query_topic` | String | The "Breakout" trend from the Google Trends CSV. |
| `source_platform` | String | Origin (NewsAPI, Reddit, Hacker News, Wikipedia). |
| `title` | String | Headline or post title. |
| `content_text` | String | The raw text used for RAG context. |
| `timestamp` | DateTime | Original publication time. |
| `engagement_score` | Integer | Upvotes (Reddit) or Points (Hacker News). |
