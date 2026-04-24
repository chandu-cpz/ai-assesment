from unittest.mock import MagicMock, patch

from section2.generator import _extract_numeric_values, answer_from_sources
from section2.pipeline import RAGPipeline
from section2.retriever import RetrievedChunk


def test_query_returns_required_shape() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What is the notice period in the NDA with Vendor X?")
    assert set(result) >= {"answer", "sources", "confidence", "answer_method"}
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["confidence"], float)
    assert result["answer_method"] in ("heuristic", "llm", "refusal")
    assert result["sources"]


def test_query_refuses_when_context_missing() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What is the arbitration seat in the reseller agreement with Vendor Z?")
    assert result == {
        "answer": "I do not have sufficient grounded context to answer that reliably.",
        "sources": [],
        "confidence": 0.0,
        "answer_method": "refusal",
    }


def test_query_prefers_contract_aligned_notice_clause() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What is the notice period in the agreement with Vendor Y?")
    assert result["answer"] == "The notice period is 45 days' written notice."
    assert result["sources"] == [
        {
            "document": "vendor_y_msa.pdf",
            "page": 4,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_refuses_when_clause_missing_for_named_contract() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What is the liability cap in the NDA with Vendor X?")
    assert result["answer"] == "I do not have sufficient grounded context to answer that reliably."
    assert result["sources"] == []
    assert result["answer_method"] == "refusal"


def test_query_handles_paraphrased_governing_law_question() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("Which state's law applies to the Vendor X NDA?")
    assert result["answer"] == "The agreement is governed by the laws of Karnataka, India."
    assert result["sources"] == [
        {
            "document": "vendor_x_nda.pdf",
            "page": 3,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_identifies_contract_above_liability_threshold() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("Which contract contains a limitation of liability clause above ₹1 crore?")
    assert result["answer"] == "The contract with a liability cap above ₹1 crore is vendor_y_msa.pdf (page 3)."
    assert result["sources"] == [
        {
            "document": "vendor_y_msa.pdf",
            "page": 3,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_uptime_commitment_with_clause_citation() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("Which clause sets the uptime commitment in the cloud hosting agreement?")
    assert result["answer"] == "The 4. Service Levels clause sets a 99.5 percent monthly uptime commitment."
    assert result["sources"] == [
        {
            "document": "cloud_hosting_agreement.pdf",
            "page": 2,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_invoice_terms_offline() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("When are invoices due under the MSA with Vendor Y?")
    assert result["answer"] == "Undisputed invoices are due within 30 days after receipt of a valid invoice."
    assert result["answer_method"] == "heuristic"
    assert result["sources"] == [
        {
            "document": "vendor_y_msa.pdf",
            "page": 2,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_service_credit_offline() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What service credit applies if uptime falls below 99.5 percent in the cloud hosting agreement?")
    assert result["answer"] == "The customer receives a 10 percent service credit of the monthly recurring fee."
    assert result["answer_method"] == "heuristic"
    assert result["sources"] == [
        {
            "document": "cloud_hosting_agreement.pdf",
            "page": 2,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_cloud_hosting_termination_notice_offline() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("How much notice is required to terminate the cloud hosting agreement for convenience?")
    assert result["answer"] == "The notice period is 60 days' written notice."
    assert result["answer_method"] == "heuristic"
    assert result["sources"] == [
        {
            "document": "cloud_hosting_agreement.pdf",
            "page": 4,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_apache_patent_termination_offline() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("What happens to patent licenses under the Apache License if you file a patent lawsuit?")
    assert result["answer"] == "If patent litigation is filed, the patent licenses terminate on the filing date."
    assert result["answer_method"] == "heuristic"
    assert result["sources"] == [
        {
            "document": "apache_license_v2.pdf",
            "page": 2,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_query_answers_apache_warranty_disclaimer_offline() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("Does the Apache License provide any warranty on the Work?")
    assert result["answer"] == "No. The Work is provided on an AS IS basis without warranties or conditions of any kind."
    assert result["answer_method"] == "heuristic"
    assert result["sources"] == [
        {
            "document": "apache_license_v2.pdf",
            "page": 4,
            "chunk": result["sources"][0]["chunk"],
        }
    ]


def test_numeric_grounding_normalizes_indian_currency_formats() -> None:
    source_values = _extract_numeric_values("INR 1,50,00,000 (one crore fifty lakh rupees)")
    answer_values = _extract_numeric_values("The liability cap is ₹1.5 crore.")
    assert "amount:15000000" in source_values
    assert "amount:15000000" in answer_values


def test_numeric_grounding_ignores_untyped_section_numbers() -> None:
    values = _extract_numeric_values("Section 4 requires 99.5% uptime.")
    assert values == {"percent:9950"}


def test_query_refuses_when_empty() -> None:
    pipeline = RAGPipeline()
    result = pipeline.query("   ")
    assert result == {
        "answer": "I do not have sufficient grounded context to answer that reliably.",
        "sources": [],
        "confidence": 0.0,
        "answer_method": "refusal",
    }


def test_llm_path_returns_grounded_answer() -> None:
    """Mock the NIM client to test the LLM → grounding path without a real API call."""
    fake_chunk = RetrievedChunk(
        document="test_contract.pdf",
        page=2,
        clause="7. Penalties",
        chunk="Late delivery incurs a penalty of INR 5,00,000 per calendar week of delay.",
        score=0.72,
    )
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="The penalty is INR 5,00,000 per week (test_contract.pdf, page 2)."))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("section2.generator._get_nim_client", return_value=mock_client), \
         patch("section2.generator.NVIDIA_API_KEY", "fake-key"):
        result = answer_from_sources("What is the late delivery penalty?", [fake_chunk])

    assert result["answer_method"] == "llm"
    assert "5,00,000" in result["answer"]
    assert result["sources"][0]["document"] == "test_contract.pdf"
    mock_client.chat.completions.create.assert_called_once()


def test_llm_path_refuses_ungrounded_answer() -> None:
    """LLM returns a numeric value not present in sources → should refuse."""
    fake_chunk = RetrievedChunk(
        document="test_contract.pdf",
        page=1,
        clause="3. Fees",
        chunk="The annual license fee is INR 10,00,000.",
        score=0.65,
    )
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="The license fee is INR 25,00,000 per year."))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("section2.generator._get_nim_client", return_value=mock_client), \
         patch("section2.generator.NVIDIA_API_KEY", "fake-key"):
        result = answer_from_sources("What is the license fee?", [fake_chunk])

    assert result["answer_method"] == "refusal"
