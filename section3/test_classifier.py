from __future__ import annotations

import time

from .classifier import TicketClassifier
from .data_builder import LABELS
from .train import model_exists, train_model


SAMPLE_TICKETS = [
    "I was billed twice for the same workspace renewal.",
    "The dashboard freezes whenever I click export.",
    "Please add scheduled weekly reports.",
    "Support closed my case without fixing anything.",
    "Can you share your W-9 form?",
    "The invoice amount is wrong after our downgrade.",
    "SSO login loops forever on Chrome.",
    "I want Slack alerts for failed jobs.",
    "Your outage communication was extremely poor.",
    "What is your support phone number?",
    "We never received the refund for the duplicate charge.",
    "Uploading the CSV causes a blank screen.",
    "Please add dark mode to the admin portal.",
    "I keep getting transferred and nobody owns the case.",
    "How do I update the account owner email?",
    "The April invoice still includes a removed add-on.",
    "Saving workflow changes does nothing.",
    "We need multi-step approvers for invoices.",
    "The last support reply was rude and dismissive.",
    "Do you have a roadmap webinar next month?",
]


def test_predictions_for_ticket_list_are_valid_and_fast() -> None:
    if not model_exists():
        train_model()
    classifier = TicketClassifier(device="cpu")
    predictions = []
    latencies_ms = []
    for ticket in SAMPLE_TICKETS:
        start = time.perf_counter()
        predictions.append(classifier.predict(ticket))
        latencies_ms.append((time.perf_counter() - start) * 1000)
    assert len(predictions) == len(SAMPLE_TICKETS)
    assert max(latencies_ms) < 500.0
    for prediction in predictions:
        assert prediction in LABELS
