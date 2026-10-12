# config.py

import os
from pathlib import Path

# Load .env file if present.
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path)
    except ImportError:
        def _load_dotenv_file(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    if key and value and key not in os.environ:
                        os.environ[key] = value

        _load_dotenv_file(env_path)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY")

if not NEWS_API_KEY:
    raise RuntimeError(
        "NEWS_API_KEY is not set. Add NEWS_API_KEY to .env or export it in the environment."
    )

if not GNEWS_API_KEY:
    raise RuntimeError(
        "GNEWS_API_KEY is not set. Add GNEWS_API_KEY to .env or export it in the environment."
    )

# 2. Keyword Buckets for Sectors (Refined from literature reviews)
KEYWORDS = {
    "healthcare": {
        "news": '("Digital Twin" OR "Virtual Twin") AND ("healthcare" OR "hospital" OR "medical") AND "sustainability"',
        "arxiv": 'all:"Digital Twin" AND (all:healthcare OR all:medical OR all:hospital)',
    },
    "manufacturing": {
        "news": '("Digital Twin" OR "Industry 4.0") AND ("manufacturing" OR "factory") AND "sustainability"',
        "arxiv": 'all:"Digital Twin" AND (all:manufacturing OR all:factory OR all:"Industry 4.0")',
    },
    "business": {
        "news": '("Digital Twin" OR "Enterprise Twin") AND ("business" OR "supply chain" OR "logistics") AND "sustainability"',
        "arxiv": 'all:"Digital Twin" AND (all:business OR all:"supply chain" OR all:logistics)',
    },
    
}

GNEWS_KEYWORDS = {
    "healthcare": "digital twin healthcare",
    "manufacturing": "digital twin manufacturing",
    "business": "digital twin logistics",
}