from __future__ import annotations

from .generator import _refusal, answer_from_sources
from .retriever import ContractRetriever, RetrievedChunk


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = ContractRetriever()

    def query(self, question: str) -> dict[str, object]:
        if not isinstance(question, str) or not question.strip():
            return _refusal(0.0)
        question = question.strip()
        sources = self.retriever.retrieve(question=question, top_k=3, candidate_k=12)
        return answer_from_sources(question, sources)

    def retrieve_sources(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question=question, top_k=top_k, candidate_k=max(top_k + 4, 12))


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print(pipeline.query("What is the notice period in the NDA with Vendor X?"))
