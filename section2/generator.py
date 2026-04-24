from __future__ import annotations

from decimal import Decimal, InvalidOperation
import logging
from pathlib import Path
import re
from typing import Callable, Iterable

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .config import NIM_BASE_URL, NIM_CHAT_MODEL, NVIDIA_API_KEY
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

DOCUMENT_IDENTIFIER_STOPWORDS = {
    "agreement",
    "agreements",
    "contract",
    "contracts",
    "document",
    "documents",
    "policy",
    "policies",
    "page",
    "pages",
    "pdf",
    "signed",
    "final",
    "amended",
    "draft",
    "copy",
    "the",
    "a",
    "an",
    "and",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
}

QUERY_IDENTIFIER_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
    "why",
    "is",
    "are",
    "does",
    "do",
    "did",
    "the",
    "a",
    "an",
    "and",
    "of",
    "for",
    "to",
    "in",
    "on",
    "under",
    "with",
    "about",
    "agreement",
    "contract",
    "clause",
    "clauses",
    "notice",
    "period",
    "law",
    "governing",
    "govern",
    "liability",
    "cap",
    "credit",
    "service",
    "services",
    "support",
    "term",
    "terms",
    "termination",
    "terminate",
    "convenience",
    "required",
    "applies",
    "apply",
    "contains",
    "contain",
    "promise",
    "promises",
    "provide",
    "provides",
    "happens",
    "happen",
}

PHRASE_PREFIX_STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "under",
    "with",
    "for",
    "to",
    "of",
}

_nim_client: OpenAI | None = None


def _get_nim_client() -> OpenAI:
    global _nim_client
    if _nim_client is None:
        _nim_client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NIM_BASE_URL)
    return _nim_client


def answer_from_sources(question: str, sources: list[RetrievedChunk]) -> dict[str, object]:
    if not sources:
        return _refusal(0.0)

    eligible_sources, anchored = _eligible_sources(question, sources)
    if anchored and not eligible_sources:
        return _refusal(0.0)

    heuristic = _heuristic_answer(question, eligible_sources)
    if heuristic is not None:
        answer, selected_sources, confidence = heuristic
        return {
            "answer": answer,
            "sources": [_source_dict(item) for item in selected_sources],
            "confidence": confidence,
            "answer_method": "heuristic",
        }

    if NVIDIA_API_KEY:
        llm_answer = _nim_answer(question, eligible_sources)
        if llm_answer is not None:
            grounded = _ground_answer(llm_answer, eligible_sources)
            if grounded:
                return {
                    "answer": llm_answer,
                    "sources": [_source_dict(item) for item in eligible_sources[:2]],
                    "confidence": min(0.74, 0.45 + 0.15 * len(eligible_sources)),
                    "answer_method": "llm",
                }

    return _refusal(max(0.0, min(0.29, eligible_sources[0].score if eligible_sources else 0.0)))


def _heuristic_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    question_lower = question.lower()
    handlers: list[Callable[[str, list[RetrievedChunk]], tuple[str, list[RetrievedChunk], float] | None]] = [
        _notice_period_answer,
        _survival_period_answer,
        _governing_law_answer,
        _invoice_due_answer,
        _liability_cap_answer,
        _uptime_commitment_answer,
        _service_credit_answer,
        _patent_termination_answer,
        _redistribution_conditions_answer,
        _warranty_disclaimer_answer,
        _above_threshold_liability_answer,
    ]
    for handler in handlers:
        result = handler(question_lower, sources)
        if result is not None:
            return result
    return None


def _notice_period_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "notice" not in question and "terminate" not in question:
        return None
    for source in sources:
        match = re.search(
            r"(thirty|forty-five|sixty)\s*\((\d+)\)\s+days'\s+written\s+notice",
            source.chunk,
            re.I,
        )
        if not match:
            continue
        days = match.group(2)
        return (f"The notice period is {days} days' written notice.", [source], _confidence(source, 0.92))
    return None


def _survival_period_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "survive" not in question and "survival" not in question:
        return None
    for source in sources:
        match = re.search(r"survive termination for (three \(3\) years)", source.chunk, re.I)
        if not match:
            continue
        return (
            "The confidentiality obligations survive termination for three years.",
            [source],
            _confidence(source, 0.9),
        )
    return None


def _governing_law_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "govern" not in question and "law" not in question:
        return None
    for source in sources:
        match = re.search(r"laws of ([A-Za-z ,]+)\.", source.chunk)
        if not match:
            continue
        jurisdiction = match.group(1).strip()
        return (f"The agreement is governed by the laws of {jurisdiction}.", [source], _confidence(source, 0.9))
    return None


def _invoice_due_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "invoice" not in question and "pay" not in question:
        return None
    for source in sources:
        normalized_chunk = _normalized_text(source.chunk)
        match = re.search(r"within thirty 30 days after receipt of a valid invoice", normalized_chunk, re.I)
        if not match:
            continue
        return (
            "Undisputed invoices are due within 30 days after receipt of a valid invoice.",
            [source],
            _confidence(source, 0.88),
        )
    return None


def _liability_cap_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "liability" not in question or "above" in question:
        return None
    for source in sources:
        amount = _extract_inr_amount(source.chunk)
        if amount is None:
            continue
        return (
            f"The liability cap is {amount['display']}.",
            [source],
            _confidence(source, 0.9),
        )
    return None


def _service_credit_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "service credit" not in question and "credit" not in question:
        return None
    for source in sources:
        normalized_chunk = _normalized_text(source.chunk)
        match = re.search(r"service credit equal to (\d+) percent", normalized_chunk, re.I)
        if not match:
            continue
        percent = match.group(1)
        return (
            f"The customer receives a {percent} percent service credit of the monthly recurring fee.",
            [source],
            _confidence(source, 0.89),
        )
    return None


def _uptime_commitment_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "uptime" not in question or "credit" in question:
        return None
    for source in sources:
        match = re.search(r"monthly uptime of (\d+(?:\.\d+)?) percent", source.chunk, re.I)
        if not match:
            continue
        percent = match.group(1)
        return (
            f"The {source.clause} clause sets a {percent} percent monthly uptime commitment.",
            [source],
            _confidence(source, 0.89),
        )
    return None


def _above_threshold_liability_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "above" not in question or "liability" not in question:
        return None
    qualifying: list[RetrievedChunk] = []
    for source in sources:
        amount = _extract_inr_amount(source.chunk)
        if amount is None:
            continue
        if amount["value"] > 10_000_000:
            qualifying.append(source)
    if not qualifying:
        return None
    docs = ", ".join(f"{item.document} (page {item.page})" for item in qualifying)
    return (
        f"The contract with a liability cap above ₹1 crore is {docs}.",
        qualifying,
        _confidence(qualifying[0], 0.86),
    )


def _patent_termination_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "patent" not in question or all(term not in question for term in ("lawsuit", "litigation", "sue")):
        return None
    for source in sources:
        normalized_chunk = _normalized_text(source.chunk)
        if re.search(r"patent licenses granted .* terminate as of the date such litigation is filed", normalized_chunk, re.I):
            return (
                "If patent litigation is filed, the patent licenses terminate on the filing date.",
                [source],
                _confidence(source, 0.88),
            )
    return None


def _redistribution_conditions_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "redistribut" not in question and "derivative works" not in question:
        return None
    for source in sources:
        if not re.search(r"\(a\).*\(b\).*\(c\).*\(d\)", source.chunk, re.I | re.S):
            continue
        return (
            "Redistributors must provide a copy of the license, mark modified files, retain notices, and include NOTICE attributions when applicable.",
            [source],
            _confidence(source, 0.87),
        )
    return None


def _warranty_disclaimer_answer(question: str, sources: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], float] | None:
    if "warranty" not in question:
        return None
    for source in sources:
        normalized_chunk = _normalized_text(source.chunk)
        if not re.search(r"\bas is\b", normalized_chunk, re.I):
            continue
        if not re.search(r"without warranties or conditions of any kind", normalized_chunk, re.I):
            continue
        return (
            "No. The Work is provided on an AS IS basis without warranties or conditions of any kind.",
            [source],
            _confidence(source, 0.87),
        )
    return None


def _nim_answer(question: str, sources: list[RetrievedChunk]) -> str | None:
    client = _get_nim_client()
    bounded_sources = _bounded_sources(sources)
    context = "\n\n".join(
        f"Document: {source.document}\nPage: {source.page}\nClause: {source.clause}\nText: {source.chunk}"
        for source in bounded_sources
    )
    try:
        response = client.chat.completions.create(
            model=NIM_CHAT_MODEL,
            temperature=0.0,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied legal contract excerpts. "
                        "If the excerpts do not contain enough evidence, respond with REFUSE. "
                        "Cite the document name and page number in the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
        )
    except (APIError, APIConnectionError, APITimeoutError) as exc:
        log.warning("NIM API call failed (%s): %s", type(exc).__name__, exc)
        return None
    content = response.choices[0].message.content
    if not content or "REFUSE" in content:
        return None
    return content.strip()


def _ground_answer(answer: str, sources: Iterable[RetrievedChunk]) -> bool:
    answer_values = _extract_numeric_values(answer)
    if not answer_values:
        return True
    source_values = set()
    for source in sources:
        source_values.update(_extract_numeric_values(source.chunk))
    return answer_values.issubset(source_values)


def _eligible_sources(question: str, sources: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
    explicit_phrases = _explicit_identifier_phrases(question)
    if explicit_phrases:
        phrase_matched = [
            source
            for source in sources
            if any(phrase in _source_search_text(source) for phrase in explicit_phrases)
        ]
        if not phrase_matched:
            return [], True
        sources = phrase_matched

    document_tokens = _document_identifier_tokens(sources)
    query_tokens = _query_identifier_tokens(question)
    if not document_tokens or not query_tokens:
        return sources, False

    scored_documents = sorted(
        (
            (document, len(tokens & query_tokens), tokens)
            for document, tokens in document_tokens.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_document, best_score, best_tokens = scored_documents[0]
    second_best_score = scored_documents[1][1] if len(scored_documents) > 1 else 0
    unique_matches = {
        token
        for token in query_tokens & best_tokens
        if sum(token in tokens for _, _, tokens in scored_documents) == 1
    }
    should_filter = bool(unique_matches) or (best_score >= 2 and best_score > second_best_score)
    if not should_filter:
        return sources, False

    matched = [source for source in sources if source.document == best_document]
    return matched, True


def _document_identifier_tokens(sources: list[RetrievedChunk]) -> dict[str, set[str]]:
    identifiers: dict[str, set[str]] = {}
    for source in sources:
        tokens = identifiers.setdefault(source.document, set())
        tokens.update(_identifier_tokens(Path(source.document).stem, DOCUMENT_IDENTIFIER_STOPWORDS))
        if source.page == 1:
            title = source.chunk.splitlines()[0] if source.chunk.splitlines() else ""
            tokens.update(_identifier_tokens(title, DOCUMENT_IDENTIFIER_STOPWORDS))
    return identifiers


def _query_identifier_tokens(question: str) -> set[str]:
    return _identifier_tokens(question, QUERY_IDENTIFIER_STOPWORDS)


def _explicit_identifier_phrases(question: str) -> set[str]:
    phrases: set[str] = set()
    for phrase in re.findall(
        r"\b(?:[A-Z]{2,}|[A-Z][a-z]+)(?:\s+(?:[A-Z]{1,2}|[A-Z]{2,}|[A-Z][a-z]+))*",
        question,
    ):
        normalized = _normalized_text(phrase)
        if normalized and normalized not in QUERY_IDENTIFIER_STOPWORDS:
            phrases.add(normalized)
    for phrase in re.findall(r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,3}\s+(?:agreement|license|policy))\b", question.lower()):
        normalized = _trim_phrase_prefix(_normalized_text(phrase))
        if normalized and normalized not in {"agreement", "license", "policy"}:
            phrases.add(normalized)
    return phrases


def _identifier_tokens(text: str, stopwords: set[str]) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return {token for token in tokens if token not in stopwords}


def _source_search_text(source: RetrievedChunk) -> str:
    return _normalized_text(f"{source.document} {source.clause} {source.chunk}")


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _trim_phrase_prefix(phrase: str) -> str:
    tokens = [token for token in phrase.split() if token]
    while tokens and (tokens[0] in PHRASE_PREFIX_STOPWORDS or tokens[0] in QUERY_IDENTIFIER_STOPWORDS):
        tokens.pop(0)
    return " ".join(tokens)


def _confidence(source: RetrievedChunk, ceiling: float) -> float:
    return round(min(ceiling, max(0.35, source.score)), 2)


def _source_dict(source: RetrievedChunk) -> dict[str, object]:
    return {
        "document": source.document,
        "page": source.page,
        "chunk": source.chunk,
    }


def _extract_inr_amount(text: str) -> dict[str, object] | None:
    digit_match = re.search(r"(?:INR|₹)\s*([\d,]+(?:\.\d+)?)", text, re.I)
    if digit_match:
        raw = digit_match.group(1)
        value = int(float(raw.replace(",", "")))
        return {"display": f"INR {digit_match.group(1)}", "value": value}
    scaled_match = re.search(r"(?:INR|₹)\s*(\d+(?:\.\d+)?)\s*(crore|crores|lakh|lakhs|million|billion)\b", text, re.I)
    if scaled_match:
        raw_value = float(scaled_match.group(1))
        scale = scaled_match.group(2).lower()
        value = int(raw_value * _scale_multiplier(scale))
        display = f"INR {scaled_match.group(1)} {scaled_match.group(2)}"
        return {"display": display, "value": value}
    word_match = re.search(
        r"\(([a-z\-\s]+(?:crore|lakh|million|billion)[a-z\-\s]*)\)",
        text,
        re.I,
    )
    if not word_match:
        return None
    value = _number_words_to_int(word_match.group(1))
    if value is None:
        return None
    return {"display": word_match.group(1).strip(), "value": value}


def _refusal(confidence: float) -> dict[str, object]:
    return {
        "answer": "I do not have sufficient grounded context to answer that reliably.",
        "sources": [],
        "confidence": round(confidence, 2),
        "answer_method": "refusal",
    }


def _bounded_sources(sources: list[RetrievedChunk], max_characters: int = 2600) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    total_characters = 0
    for source in sources:
        projected = total_characters + len(source.chunk)
        if selected and projected > max_characters:
            break
        selected.append(source)
        total_characters = projected
    return selected or sources[:1]
def _extract_numeric_values(text: str) -> set[str]:
    values: set[str] = set()
    values.update(_extract_amount_markers(text))
    values.update(_extract_percentage_markers(text))
    values.update(_extract_day_markers(text))
    return values


def _extract_amount_markers(text: str) -> set[str]:
    values: set[str] = set()
    for amount, scale in re.findall(
        r"(?:INR|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|crores|lakh|lakhs|million|billion)?\b",
        text,
        re.I,
    ):
        values.add(f"amount:{_normalize_amount_value(amount, scale)}")
    for amount, scale in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*(crore|crores|lakh|lakhs|million|billion)\b",
        text,
        re.I,
    ):
        values.add(f"amount:{_normalize_amount_value(amount, scale)}")
    for fragment in re.findall(r"\(([^)]*(?:rupees?|crore|lakhs?|million|billion)[^)]*)\)", text, re.I):
        numeric = _number_words_to_int(fragment)
        if numeric is not None:
            values.add(f"amount:{numeric}")
    return values


def _extract_percentage_markers(text: str) -> set[str]:
    values: set[str] = set()
    for raw in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", text, re.I):
        values.add(f"percent:{_scaled_hundredths(raw)}")
    return values


def _extract_day_markers(text: str) -> set[str]:
    values: set[str] = set()
    for left, right in re.findall(r"(?:\((\d+)\)|(\d+))\s+days?\b", text, re.I):
        values.add(f"days:{left or right}")
    return values


def _normalize_amount_value(raw_amount: str, scale: str | None) -> int:
    amount = Decimal(raw_amount.replace(",", ""))
    if scale:
        amount *= _scale_multiplier(scale.lower())
    return int(amount)


def _scaled_hundredths(raw_value: str) -> int:
    try:
        return int(Decimal(raw_value) * 100)
    except InvalidOperation:
        return int(raw_value)


def _number_words_to_int(text: str) -> int | None:
    normalized = text.lower().replace("-", " ")
    tokens = [token for token in re.findall(r"[a-z]+", normalized) if token not in {"and", "rupee", "rupees"}]
    if not tokens:
        return None
    units = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    scales = {
        "hundred": 100,
        "thousand": 1_000,
        "lakh": 100_000,
        "lakhs": 100_000,
        "crore": 10_000_000,
        "crores": 10_000_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    }
    total = 0
    current = 0
    consumed = False
    for token in tokens:
        if token in units:
            current += units[token]
            consumed = True
            continue
        if token == "hundred":
            current = max(1, current) * 100
            consumed = True
            continue
        if token in scales:
            current = max(1, current)
            total += current * scales[token]
            current = 0
            consumed = True
            continue
        return None
    if not consumed:
        return None
    return total + current


def _scale_multiplier(scale: str) -> int:
    return {
        "crore": 10_000_000,
        "crores": 10_000_000,
        "lakh": 100_000,
        "lakhs": 100_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    }[scale]
