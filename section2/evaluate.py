from __future__ import annotations

import json

from .config import EVAL_QA_PATH
from .ingest import export_eval_set
from .pipeline import RAGPipeline


def evaluate_precision_at_3() -> dict[str, object]:
    export_eval_set()
    eval_rows = json.loads(EVAL_QA_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    retrieval_rows = [row for row in eval_rows if row["kind"] == "retrieval"]
    refusal_rows = [row for row in eval_rows if row["kind"] == "refusal"]

    retrieval_hits = 0
    retrieval_details: list[dict[str, object]] = []
    by_type: dict[str, dict[str, int]] = {}
    for row in retrieval_rows:
        full_result = pipeline.query(row["question"])
        retrieved = pipeline.retrieve_sources(row["question"], top_k=3)
        matched_item = next(
            (
                (index, item)
                for index, item in enumerate(retrieved, start=1)
                if item.document == row["expected_document"] and item.page == row["expected_page"]
            ),
            None,
        )
        hit = matched_item is not None
        retrieval_hits += int(hit)
        stats = by_type.setdefault(row["question_type"], {"hits": 0, "total": 0})
        stats["hits"] += int(hit)
        stats["total"] += 1
        retrieval_details.append(
            {
                "question": row["question"],
                "kind": row["kind"],
                "question_type": row["question_type"],
                "expected_answer": row.get("expected_answer"),
                "returned_answer": full_result.get("answer"),
                "hit": hit,
                "matched_rank": matched_item[0] if matched_item else None,
                "matched_document": matched_item[1].document if matched_item else None,
                "matched_page": matched_item[1].page if matched_item else None,
                "answer_method": full_result.get("answer_method", "unknown"),
                "retrieved": [
                    {"document": item.document, "page": item.page, "score": round(item.score, 3)}
                    for item in retrieved
                ],
            }
        )

    refusal_hits = 0
    refusal_details: list[dict[str, object]] = []
    for row in refusal_rows:
        result = pipeline.query(row["question"])
        refused = not result["sources"] and result["confidence"] < 0.3
        refusal_hits += int(refused)
        refusal_details.append(
            {
                "question": row["question"],
                "kind": row["kind"],
                "question_type": row["question_type"],
                "hit": refused,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "answer_method": result.get("answer_method", "unknown"),
            }
        )

    precision = retrieval_hits / len(retrieval_rows)
    refusal_accuracy = refusal_hits / len(refusal_rows) if refusal_rows else 0.0
    report = {
        "precision_at_3": precision,
        "hits": retrieval_hits,
        "total": len(retrieval_rows),
        "refusal_accuracy": refusal_accuracy,
        "refusal_hits": refusal_hits,
        "refusal_total": len(refusal_rows),
        "by_question_type": by_type,
        "details": retrieval_details + refusal_details,
    }
    report_path = EVAL_QA_PATH.with_name("eval_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    report = evaluate_precision_at_3()
    print(f"Precision@3: {report['precision_at_3']:.2%} ({report['hits']}/{report['total']})")
    if report["refusal_total"]:
        print(
            f"Refusal accuracy: {report['refusal_accuracy']:.2%} "
            f"({report['refusal_hits']}/{report['refusal_total']})"
        )
    print("Question-type breakdown:")
    for question_type, stats in sorted(report["by_question_type"].items()):
        print(f"- {question_type}: {stats['hits']}/{stats['total']} ({stats['hits'] / stats['total']:.2%})")
    for item in report["details"]:
        status = "PASS" if item["hit"] else "FAIL"
        if item["kind"] == "retrieval":
            if item["hit"]:
                print(
                    f"- {status} [{item['question_type']}]: {item['question']} "
                    f"-> {item['matched_document']} page {item['matched_page']} "
                    f"(rank {item['matched_rank']})"
                )
            else:
                top = item["retrieved"][0] if item["retrieved"] else {"document": "n/a", "page": "n/a"}
                print(
                    f"- {status} [{item['question_type']}]: {item['question']} "
                    f"-> top hit {top['document']} page {top['page']}"
                )
            continue
        print(
            f"- {status} [{item['question_type']}]: {item['question']} "
            f"-> confidence {item['confidence']:.2f}"
        )
    print(f"Saved evaluation report to {report['report_path']}")


if __name__ == "__main__":
    main()
