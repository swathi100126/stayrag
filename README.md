# StayRAG

StayRAG is a Streamlit PG discovery demo with city, budget, room, meals, and stay-preference filters; side-by-side comparisons; and a conversational assistant grounded in the included listing catalogue.

## Run locally

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Optional: copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. The app uses the official Groq Python SDK and defaults to `llama-3.3-70b-versatile`. Without a key, StayRAG stays usable in deterministic, catalogue-only mode.

3. Start the app:

   ```powershell
   streamlit run app.py
   ```

Run the retrieval tests with:

```powershell
python -m unittest discover -s tests
```

## Data and grounding

The sample catalogue is `data/pg_listings.csv`. StayRAG validates its required columns, builds TF-IDF vectors from listing facts, and ranks matches with cosine similarity. Groq answers are constrained to retrieved listing context; when details are missing, the assistant is instructed to say so. The included stays are illustrative demo data, not live listings or availability. Confirm prices, rules, and availability with a property before booking.
