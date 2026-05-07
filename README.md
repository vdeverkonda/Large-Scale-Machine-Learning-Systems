# Bank of America Research Engineer Portfolio Projects

## Projects

1. **Time-Series Forecasting System**
   - Forecasts synthetic payment transaction volume and revenue.
   - Uses lag features, rolling statistics, random forest, and optional LSTM extension.

2. **A/B Testing and Experimentation Framework**
   - Simulates control/treatment experiments.
   - Computes lift, p-values, confidence intervals, power, and business impact.

3. **RAG Knowledge Search System**
   - Builds a retrieval-augmented generation style search pipeline using TF-IDF embeddings.
   - Designed so it can later be upgraded to FAISS, Pinecone, LangChain, or OpenAI embeddings.

4. **Dynamic Pricing Optimization Model**
   - Simulates price-demand relationships.
   - Estimates demand curve and finds revenue-maximizing price.

5. **Real-Time NLP Insights Engine**
   - Simulates servicing/client support messages.
   - Uses NLP topic extraction and sentiment analysis to identify near real-time client concerns.

## How to Run

Each project has its own README and script.

```bash
cd 01_time_series_forecasting
pip install -r requirements.txt
python forecasting_pipeline.py
```

