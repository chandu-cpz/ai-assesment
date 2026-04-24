# Section 2 Design: Legal-Document RAG Pipeline

## Goal and Constraints

The target workload is precise contract QA over a small but non-trivial corpus of legal PDFs. Users ask clause-level questions such as notice periods, liability caps, and governing law. Hallucinated answers are unacceptable, and every answer must cite the exact source document and page. That changes the design objective: high precision and verifiable grounding matter more than stylistic fluency.

For the assessment repository, this is implemented as a locally runnable prototype over four bundled PDFs. The prototype is intentionally small so a reviewer can run it from a clean checkout, but the design choices are the same ones I would keep when scaling the system up.

## Chunking Strategy

I used section-aware chunking with a size cap rather than naive fixed windows. The ingestion step first preserves page boundaries, then splits on blank lines and clause numbering so that sections such as `8. Limitation of Liability` stay intact when possible. Chunks are capped at roughly 900 characters with a 120-character overlap. The overlap is deliberate because legal obligations often span two sentences: the first introduces the rule and the second adds carve-outs or exceptions. A hard cut with no overlap tends to separate the cap from the exception language, which hurts recall for questions like liability carve-outs.

I did not use extremely large chunks because they dilute retrieval. A contract question such as “What is the notice period in the NDA with Vendor X?” should retrieve the clause, not an entire page plus unrelated confidentiality and governing-law text.

## Embedding Choice

For the runnable repository I use `sentence-transformers/all-MiniLM-L6-v2` as the default embedding backend. The reason is practical: it keeps setup local, deterministic, and fast enough to satisfy the “clean environment” requirement without relying on an external embedding service. For a production version of the same pipeline I would switch the embedding backend to NVIDIA NIM `nvidia/nv-embedqa-e5-v5`, because it is tuned for retrieval-style semantic search and is a better fit for dense clause lookup than a general-purpose small sentence encoder.

- local MiniLM: lower operational friction, predictable setup, slightly weaker semantic recall
- NIM `nv-embedqa-e5-v5`: stronger retrieval quality, higher operational dependency, external API latency/cost

## Vector Store Choice

I chose ChromaDB instead of FAISS or Pinecone.

- ChromaDB gives me local persistence, metadata filtering, and a simple API without needing managed infrastructure.
- FAISS is strong for approximate nearest-neighbor search, but for this assessment it would require more custom persistence and metadata plumbing.
- Pinecone would be more appropriate if I needed multi-tenant serving, managed scaling, and operational SLAs, but that is overkill for a locally runnable submission.

For a corpus of roughly 500 contracts averaging 40 pages each, Chroma is completely adequate. The index remains small enough to build locally, and the metadata model is useful because the retrieval layer can expose source filename and page directly.

I also added a small operational guardrail that matters in practice: the retriever stores a content signature for the chunk manifest beside the Chroma collection and forces a rebuild when the corpus changes. Counting rows alone is not enough because legal documents are often revised in place without changing the number of chunks.

## Retrieval Strategy

I implemented a two-stage retrieval flow.

1. Dense retrieval from Chroma returns the top semantic candidates.
2. A lightweight clause reranker then reorders those results with lexical overlap and entity boosts.

The reranker exists because legal questions are often entity-sensitive. A semantic search may find a clause about “termination notice” in the wrong contract if multiple documents talk about termination. The reranker boosts chunks that explicitly mention the contract name or the query’s key tokens. In a production version I would replace that light reranker with NVIDIA NIM reranking (`nvidia/nv-rerankqa-mistral-4b-v3`) because cross-encoder reranking usually improves precision materially on dense legal text.

The current reranker is metadata-aware rather than hardcoded to the sample contracts. It scores semantic similarity first, then adds generic overlap and phrase-alignment signals from the document name and clause header. That keeps the local repository simple while avoiding a common assessment anti-pattern where retrieval logic is quietly overfit to the exact filenames in the demo set.

I did not use naive top-k alone because top-k cosine retrieval is too willing to surface adjacent but wrong clauses. For legal QA, “close but wrong” is not acceptable.

## Hallucination Mitigation

The repository implements three safeguards:

1. Source-grounded answer composition. The answer builder prefers deterministic extraction from retrieved chunks. If it cannot find a clause-supported answer, it refuses.
2. Confidence gating. Confidence combines retrieval rank and whether the answer’s numeric or contractual terms can be traced back into the source chunk. Numeric checks normalize Indian currency formats such as `INR 1,50,00,000`, `₹1.5 crore`, and written forms like `one crore fifty lakh rupees` before the pipeline accepts a generated answer.
3. Bounded LLM fallback. If NVIDIA NIM is enabled, the pipeline trims context before generation and treats API errors as a refusal path rather than letting optional model failures break the query flow.

This design is conservative. A legal assistant that refuses occasionally is acceptable. One that invents a liability cap is not.

## Scaling to 50,000 Documents

At 50,000 documents, different bottlenecks appear.

1. Embedding throughput becomes the first ingestion bottleneck.
   - Remedy: move to batched asynchronous ingestion with a job queue; precompute embeddings in workers; checkpoint partial progress.

2. Local Chroma on a single machine becomes a retrieval and storage bottleneck.
   - Remedy: move to a distributed vector store such as Qdrant, Weaviate, or Pinecone; partition by tenant or document family; use metadata filtering aggressively before reranking.

3. Reranking latency grows if every query reranks too many candidates.
   - Remedy: introduce a cheaper first-stage filter using metadata or sparse search; rerank only the top 20 or fewer candidates; cache frequent queries.

4. Document freshness and traceability become harder.
   - Remedy: add versioned document IDs, ingestion manifests, and a reindex policy that preserves old citations for auditability.

5. Evaluation drift becomes more important.
   - Remedy: expand the golden set across contract types, track precision@k and answer refusal rate over time, and review failures by question type rather than only aggregate score.

## LLM Backend Choice

The assessment specification says to "use OpenAI or a free-tier alternative." This submission uses NVIDIA NIM as the LLM backend. NIM exposes an OpenAI-compatible chat completions API and provides free-tier API credits, making it a direct free-tier alternative. The `openai` Python client is used throughout; switching to OpenAI proper requires only changing the base URL and API key in `.env`. The pipeline also runs fully without any API key — the heuristic answer path and refusal logic operate locally with zero external calls.

## Evaluation Limitations

The reported precision@3 of 100% and refusal accuracy of 100% are measured on four bundled PDFs in the repository: three generated contract documents plus the Apache 2.0 license text. This is intentional for reproducibility: any reviewer can run the evaluation from a clean checkout without sourcing real legal documents. However, these scores should not be interpreted as production-quality recall numbers.

The answer generation path is deliberately conservative and prototype-shaped. Retrieval and document selection are generic, but the deterministic extraction handlers are tuned to the clause patterns present in the bundled documents so the submission can run offline and refuse cleanly when support is missing. On real legal PDFs with OCR noise, inconsistent formatting, dense nested clauses, and varied contract styles, both retrieval precision and answer extraction accuracy would be lower.

The evaluation report includes `expected_answer`, `returned_answer`, and `answer_method` for each question, making it transparent whether a given answer came from the deterministic heuristic path, the LLM fallback, or was refused. On the current bundled evaluation set, the pipeline can answer or refuse fully offline, so the optional LLM path is best understood as extra coverage for future document types rather than a requirement for the demo corpus.

To test the pipeline on documents the heuristic handlers were not specifically written for, the recommended approach is to add real public-domain contracts to `section2/data/pdfs/` and extend `section2/data/eval_qa.json` with new question–answer pairs.
