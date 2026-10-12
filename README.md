# DigitalTwin Project

## Run from a single `main.py`

Yes — `main.py` is the project entrypoint and executes the full data collection and analysis pipeline.

## Requirements

1. Python 3.10+ installed.
2. Create and activate a virtual environment.
3. Install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

4. Download the spaCy English model:

```powershell
python -m spacy download en_core_web_sm
```

5. Create a `.env` file from `.env.example` and add your API keys.

```powershell
copy .env.example .env
```

Then edit `.env` and set:

```text
NEWS_API_KEY=your_news_api_key_here
GNEWS_API_KEY=your_gnews_api_key_here
```

## Run the project

```powershell
python main.py
```

## Notes

- The pipeline starts by fetching data from APIs and then processes it into output CSVs.
- If optional packages like `transformers`, `torch`, or `bertopic` are missing, the advanced AI steps will skip with a warning message, but the rest of the pipeline can still run.
- `generate_reference.py` will create `data_sources_reference.csv` and print a preview table.
