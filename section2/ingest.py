from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import logging

import fitz
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EVAL_QA_PATH, PDF_DIR

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document: str
    page: int
    clause: str
    text: str


SAMPLE_CONTRACTS = [
    {
        "filename": "vendor_x_nda.pdf",
        "title": "Mutual Non-Disclosure Agreement with Vendor X",
        "pages": [
            [
                "Mutual Non-Disclosure Agreement with Vendor X",
                "Effective Date: 14 January 2026.",
                "Parties: Apex Devices India Private Limited and Vendor X Analytics Pvt. Ltd.",
                "Purpose: evaluating a proposed data-sharing engagement for chip design support.",
            ],
            [
                "2. Confidentiality Obligations",
                "Each receiving party shall protect Confidential Information using at least reasonable care and may use it only for the permitted purpose.",
                "4. Survival",
                "The confidentiality obligations in this Agreement survive termination for three (3) years from the effective date of termination.",
            ],
            [
                "7. Term and Termination",
                "Either party may terminate this Agreement for convenience by giving thirty (30) days' written notice to the other party.",
                "9. Governing Law",
                "This Agreement is governed by and construed in accordance with the laws of Karnataka, India.",
            ],
        ],
    },
    {
        "filename": "vendor_y_msa.pdf",
        "title": "Master Services Agreement with Vendor Y",
        "pages": [
            [
                "Master Services Agreement with Vendor Y",
                "Effective Date: 1 February 2026.",
                "Services: integration, deployment, and managed support for procurement analytics.",
            ],
            [
                "5. Fees and Payment",
                "Vendor Y shall invoice monthly in arrears.",
                "Customer shall pay undisputed amounts within thirty (30) days after receipt of a valid invoice.",
            ],
            [
                "11. Limitation of Liability",
                "Except for fraud, wilful misconduct, and breaches of confidentiality, each party's aggregate liability under this Agreement will not exceed INR 1,50,00,000 (one crore fifty lakh rupees).",
            ],
            [
                "14. Termination",
                "Either party may terminate this Agreement for convenience upon forty-five (45) days' written notice to the other party.",
                "15. Notice",
                "All notices must be in writing and sent to the contract managers listed in Schedule 1.",
            ],
        ],
    },
    {
        "filename": "cloud_hosting_agreement.pdf",
        "title": "Cloud Hosting and Support Agreement",
        "pages": [
            [
                "Cloud Hosting and Support Agreement",
                "Effective Date: 10 March 2026.",
                "Provider: Nimbus Hosted Systems Private Limited.",
                "Customer: Apex Devices India Private Limited.",
            ],
            [
                "4. Service Levels",
                "Provider will maintain monthly uptime of 99.5 percent.",
                "If monthly uptime falls below 99.5 percent, Customer will receive a service credit equal to 10 percent of the monthly recurring fee for the affected month.",
            ],
            [
                "8. Limitation of Liability",
                "Except for data protection breaches and unpaid fees, Provider's aggregate liability under this Agreement will not exceed INR 75,00,000 (seventy-five lakh rupees).",
            ],
            [
                "12. Termination for Convenience",
                "Customer may terminate this Agreement for convenience on sixty (60) days' written notice.",
            ],
        ],
    },
    {
        "filename": "apache_license_v2.pdf",
        "title": "Apache License, Version 2.0",
        "pages": [
            [
                "Apache License",
                "Version 2.0, January 2004",
                "http://www.apache.org/licenses/",
                "",
                "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
                "",
                '1. Definitions.',
                '"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.',
                '"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.',
                '"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.',
                '"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.',
                '"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.',
                '"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.',
                '"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work.',
                '"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship.',
                '"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner.',
                '"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.',
            ],
            [
                "2. Grant of Copyright License.",
                "Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.",
                "",
                "3. Grant of Patent License.",
                "Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted.",
                "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
            ],
            [
                "4. Redistribution.",
                "You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:",
                "(a) You must give any other recipients of the Work or Derivative Works a copy of this License; and",
                "(b) You must cause any modified files to carry prominent notices stating that You changed the files; and",
                "(c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and",
                "(d) If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file.",
                "",
                "5. Submission of Contributions.",
                "Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions.",
            ],
            [
                "6. Trademarks.",
                "This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.",
                "",
                "7. Disclaimer of Warranty.",
                'Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.',
                "",
                "8. Limitation of Liability.",
                "In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.",
                "",
                "9. Accepting Warranty or Additional Liability.",
                "While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.",
            ],
        ],
    },
]


EVAL_QA = [
    {
        "question": "What is the notice period in the NDA with Vendor X?",
        "kind": "retrieval",
        "question_type": "notice_period",
        "expected_answer": "30 days' written notice.",
        "expected_document": "vendor_x_nda.pdf",
        "expected_page": 3,
    },
    {
        "question": "How long do confidentiality obligations survive termination in the NDA with Vendor X?",
        "kind": "retrieval",
        "question_type": "survival_clause",
        "expected_answer": "The confidentiality obligations survive termination for three years.",
        "expected_document": "vendor_x_nda.pdf",
        "expected_page": 2,
    },
    {
        "question": "Which law governs the NDA with Vendor X?",
        "kind": "retrieval",
        "question_type": "governing_law",
        "expected_answer": "Karnataka, India.",
        "expected_document": "vendor_x_nda.pdf",
        "expected_page": 3,
    },
    {
        "question": "When are invoices due under the MSA with Vendor Y?",
        "kind": "retrieval",
        "question_type": "invoice_terms",
        "expected_answer": "Undisputed invoices are due within 30 days after receipt of a valid invoice.",
        "expected_document": "vendor_y_msa.pdf",
        "expected_page": 2,
    },
    {
        "question": "What is the limitation of liability cap in the MSA with Vendor Y?",
        "kind": "retrieval",
        "question_type": "liability_cap",
        "expected_answer": "INR 1,50,00,000.",
        "expected_document": "vendor_y_msa.pdf",
        "expected_page": 3,
    },
    {
        "question": "What notice is required to terminate the MSA with Vendor Y for convenience?",
        "kind": "retrieval",
        "question_type": "notice_period",
        "expected_answer": "45 days' written notice.",
        "expected_document": "vendor_y_msa.pdf",
        "expected_page": 4,
    },
    {
        "question": "What service credit applies if uptime falls below 99.5 percent in the cloud hosting agreement?",
        "kind": "retrieval",
        "question_type": "service_credit",
        "expected_answer": "A 10 percent credit of the monthly recurring fee.",
        "expected_document": "cloud_hosting_agreement.pdf",
        "expected_page": 2,
    },
    {
        "question": "Which clause sets the uptime commitment in the cloud hosting agreement?",
        "kind": "retrieval",
        "question_type": "uptime_commitment",
        "expected_answer": "Clause 4. Service Levels sets a 99.5 percent monthly uptime commitment.",
        "expected_document": "cloud_hosting_agreement.pdf",
        "expected_page": 2,
    },
    {
        "question": "What is the liability cap in the cloud hosting agreement?",
        "kind": "retrieval",
        "question_type": "liability_cap",
        "expected_answer": "INR 75,00,000.",
        "expected_document": "cloud_hosting_agreement.pdf",
        "expected_page": 3,
    },
    {
        "question": "How much notice is required to terminate the cloud hosting agreement for convenience?",
        "kind": "retrieval",
        "question_type": "notice_period",
        "expected_answer": "60 days' written notice.",
        "expected_document": "cloud_hosting_agreement.pdf",
        "expected_page": 4,
    },
    {
        "question": "What happens to patent licenses under the Apache License if you file a patent lawsuit?",
        "kind": "retrieval",
        "question_type": "patent_termination",
        "expected_answer": "Patent licenses terminate on the date the litigation is filed.",
        "expected_document": "apache_license_v2.pdf",
        "expected_page": 2,
    },
    {
        "question": "What conditions must be met when redistributing Derivative Works under the Apache License?",
        "kind": "retrieval",
        "question_type": "redistribution_conditions",
        "expected_answer": "Provide a copy of the license, mark modified files, retain notices, and include NOTICE attributions when applicable.",
        "expected_document": "apache_license_v2.pdf",
        "expected_page": 3,
    },
    {
        "question": "Does the Apache License provide any warranty on the Work?",
        "kind": "retrieval",
        "question_type": "warranty_disclaimer",
        "expected_answer": "No. The Work is provided on an AS IS basis without warranties or conditions of any kind.",
        "expected_document": "apache_license_v2.pdf",
        "expected_page": 4,
    },
    {
        "question": "What is the arbitration seat in the reseller agreement with Vendor Z?",
        "kind": "refusal",
        "question_type": "unsupported_contract",
    },
    {
        "question": "What is the liability cap in the NDA with Vendor X?",
        "kind": "refusal",
        "question_type": "unsupported_clause",
    },
    {
        "question": "Does the MSA with Vendor Y promise service credits?",
        "kind": "refusal",
        "question_type": "cross_contract_mismatch",
    },
    {
        "question": "What is the renewal term of the cloud hosting agreement?",
        "kind": "refusal",
        "question_type": "missing_clause",
    },
]


def ensure_sample_pdfs(output_dir: Path = PDF_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for contract in SAMPLE_CONTRACTS:
        path = output_dir / contract["filename"]
        if path.exists():
            continue
        pdf = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        for page_lines in contract["pages"]:
            text_object = pdf.beginText(48, height - 64)
            text_object.setFont("Helvetica", 11)
            for line in page_lines:
                wrapped_lines = _wrap_line(line, 92)
                for wrapped in wrapped_lines:
                    text_object.textLine(wrapped)
                text_object.textLine("")
            pdf.drawText(text_object)
            pdf.showPage()
        pdf.save()

    _sync_eval_set()


def _wrap_line(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def extract_chunks(pdf_dir: Path = PDF_DIR) -> list[ChunkRecord]:
    ensure_sample_pdfs(pdf_dir)
    chunks: list[ChunkRecord] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        document_name = pdf_path.name
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            log.warning("Skipping unreadable PDF %s: %s", document_name, exc)
            continue
        with doc:
            for page_index, page in enumerate(doc, start=1):
                raw_text = page.get_text("text").strip()
                if not raw_text:
                    continue
                page_chunks = _split_page_text(raw_text)
                for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                    clause = _clause_name(chunk_text)
                    chunk_id = f"{document_name}:p{page_index}:c{chunk_idx}"
                    chunks.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            document=document_name,
                            page=page_index,
                            clause=clause,
                            text=chunk_text,
                        )
                    )
    return chunks


def export_eval_set() -> list[dict[str, object]]:
    ensure_sample_pdfs()
    _sync_eval_set()
    return EVAL_QA


def _sync_eval_set() -> None:
    serialized = json.dumps(EVAL_QA, indent=2, ensure_ascii=False)
    if not EVAL_QA_PATH.exists() or EVAL_QA_PATH.read_text(encoding="utf-8") != serialized:
        EVAL_QA_PATH.write_text(serialized, encoding="utf-8")


def _split_page_text(raw_text: str) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", raw_text) if segment.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = paragraphs[0]
    for paragraph in paragraphs[1:]:
        if len(current) + len(paragraph) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{paragraph}"
            continue
        chunks.append(current)
        overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
        current = f"{overlap}\n\n{paragraph}".strip()
    chunks.append(current)
    return chunks


def _clause_name(text: str) -> str:
    first_line = text.splitlines()[0].strip()
    if first_line:
        return first_line
    return "Unlabeled Clause"
