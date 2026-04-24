from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
SECTION2_DIR = ROOT_DIR / "section2"
DATA_DIR = SECTION2_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_QA_PATH = DATA_DIR / "eval_qa.json"

COLLECTION_NAME = "legal_contracts"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_CHAT_MODEL = os.getenv("NIM_CHAT_MODEL", "meta/llama-3.3-70b-instruct")
