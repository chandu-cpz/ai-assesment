# AI Systems Assessment

This repository implements all four required sections of the assessment:

1. Section 1: diagnosis logs for a failing LLM support pipeline
2. Section 2: a runnable legal-document RAG pipeline with citation grounding and evaluation
3. Section 3: a CPU-friendly support ticket classifier using a fine-tuned small transformer
4. Section 4: written systems design answers for all three prompts

This is a submission prototype rather than a deployable product. The code is optimized for local reproducibility, conservative grounding behavior, and clear evaluation artifacts instead of broad real-world coverage claims.

The repository runs from a clean local environment. Sample legal PDFs are generated automatically on first use, so there are no binary assets to manage.

## Quick Start

Python 3.12+ is the target runtime. Approximate timing from a clean checkout:

- `uv sync --extra dev`: ~60 seconds (first run downloads PyTorch ~2 GB)
- First model downloads: ~30 seconds (`all-MiniLM-L6-v2` ~80 MB, `distilbert-base-uncased` ~250 MB)
- Classifier training: ~30 seconds
- Each evaluation harness: ~10 seconds

```bash
cd ~/git/ai-assessment
uv sync --extra dev
cp .env.example .env
```

## API Key Configuration

The assessment specification requires "OpenAI or a free-tier alternative." This submission uses **NVIDIA NIM** as an OpenAI-compatible free-tier alternative. The `openai` Python client is used throughout; switching to OpenAI proper requires only changing the base URL and API key.

**Without an API key**: The Section 2 pipeline runs fully offline using deterministic heuristic answer extraction and refusal logic. The bundled evaluation set completes with no external calls. Section 3 (classifier) requires no API key at all.

**With an API key**: If `NVIDIA_API_KEY` is set in `.env`, the Section 2 pipeline additionally uses NIM for LLM-based answer generation on questions that fall outside the heuristic handlers, with source grounding checks before accepting generated answers.

```bash
# Optional — edit .env and add your NVIDIA NIM key:
# NVIDIA_API_KEY=nvapi-...
# NIM_CHAT_MODEL=meta/llama-3.3-70b-instruct
```

Expected first-run downloads:

- `all-MiniLM-L6-v2` for retrieval embeddings
- `distilbert-base-uncased` for the classifier

## Section 2: RAG Pipeline

Run the retrieval evaluation harness:

```bash
uv run python -m section2.evaluate
```

Run a live query from Python:

```python
from section2.pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("What is the notice period in the NDA with Vendor X?")
print(result)
```

The pipeline returns:

```python
{
    "answer": str,
    "sources": [
        {
            "document": str,
            "page": int,
            "chunk": str,
        }
    ],
    "confidence": float,
    "answer_method": str,  # "heuristic", "llm", or "refusal"
}
```

## Section 3: Ticket Classifier

Generate the synthetic training set and curated evaluation set:

```bash
uv run python -m section3.generate_data
```

Train the classifier:

```bash
uv run python -m section3.train
```

Evaluate the classifier:

```bash
uv run python -m section3.evaluate
```

Run the latency assertion test:

```bash
uv run pytest section3/test_classifier.py
```

Observed results from the latest local evaluation run (`section3/data/eval_report.json` will vary slightly by machine and rerun):

- Section 2 retrieval precision@3: `100.00% (13/13)` on the bundled four-document evaluation set
- Section 2 refusal accuracy: `100.00% (4/4)`
- Section 3 evaluation accuracy: `90.00%`
- Section 3 most-confused class pair: `technical_issue` vs `complaint`
- Section 3 warm single-ticket CPU latency: `p50 6.94 ms`, `p95 10.85 ms`, `max 13.57 ms`
- Section 3 classifier startup on CPU: `887.79 ms`

Both evaluation entry points also persist machine-readable artifacts:

- `section2/data/eval_report.json`
- `section3/data/eval_report.json`

## Writing

- Architectural decisions for Section 2 are in `DESIGN.md`
- Written answers for Sections 1, 3, and 4 are in `ANSWERS.md`
- The Section 2 report includes `expected_answer`, `returned_answer`, and `answer_method` fields for transparent review

## Optional Recording

If you record the optional walkthrough, add the Loom link here before submission.

## Suggested Run Order

```bash
uv run python -m section3.generate_data
uv run python -m section3.train
uv run python -m section2.evaluate
uv run python -m section3.evaluate
uv run pytest tests/test_section2.py section3/test_classifier.py
```

## File Map

- `DESIGN.md`: Section 2 architecture and scaling discussion
- `ANSWERS.md`: Sections 1, 3, and 4 written reasoning
- `section2/`: legal-document RAG pipeline
- `section3/`: support-ticket classifier

