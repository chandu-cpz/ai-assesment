from __future__ import annotations

import json
import random
from pathlib import Path

LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

SECTION3_DIR = Path(__file__).resolve().parent
DATA_DIR = SECTION3_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.json"
EVAL_PATH = DATA_DIR / "eval.json"
MODEL_DIR = SECTION3_DIR / "models" / "ticket_classifier"
MODEL_NAME = "distilbert-base-uncased"

SEEDS = {
    "billing": 11,
    "technical_issue": 17,
    "feature_request": 23,
    "complaint": 29,
    "other": 31,
}

BILLING_TEMPLATES = [
    "{opening} I was {issue} for {product}. {closing}",
    "{opening} The invoice for {product} shows {issue}. {closing}",
    "{opening} Please help because {issue} on my {product} bill. {closing}",
]

TECHNICAL_TEMPLATES = [
    "{opening} {surface} {symptom} when I {action}. {closing}",
    "{opening} Every time I {action}, {surface} {symptom}. {closing}",
    "{opening} The {surface} {symptom} during {action}. {closing}",
]

FEATURE_TEMPLATES = [
    "{opening} It would help if the product had {request}. {closing}",
    "{opening} Please add {request} for {surface}. {closing}",
    "{opening} Our team wants {request} in the {surface}. {closing}",
]

COMPLAINT_TEMPLATES = [
    "{opening} I'm frustrated because {service_issue}. {closing}",
    "{opening} {service_issue}. This experience has been poor. {closing}",
    "{opening} I need to complain that {service_issue}. {closing}",
]

OTHER_TEMPLATES = [
    "{opening} I need {request}. {closing}",
    "{opening} Can you share {request}? {closing}",
    "{opening} I'm reaching out about {request}. {closing}",
]

OPENINGS = [
    "Hi team,",
    "Hello,",
    "Good morning,",
    "Hi support,",
    "Dear support,",
]

CLOSINGS = [
    "Please fix this soon.",
    "Thanks for checking.",
    "I'd appreciate a quick response.",
    "Please let me know what happened.",
    "This is time sensitive.",
]

BILLING_ISSUES = [
    "charged twice",
    "billed after cancellation",
    "tax calculated incorrectly",
    "missing the expected credit",
    "showing the wrong seat count",
    "applying the annual rate instead of the monthly rate",
    "still including a removed add-on",
    "using last month's amount instead of the new plan price",
]

BILLING_PRODUCTS = [
    "March subscription",
    "April renewal",
    "enterprise workspace",
    "analytics add-on",
    "monthly API package",
    "extra storage plan",
    "business tier renewal",
    "multi-seat workspace",
]

SURFACES = [
    "web app",
    "dashboard",
    "CSV export",
    "SSO login",
    "mobile app",
    "admin portal",
    "public API",
    "billing page",
]

TECHNICAL_SYMPTOMS = [
    "crashes",
    "spins forever",
    "returns a 500 error",
    "shows a blank screen",
    "fails silently",
    "logs me out",
    "never finishes loading",
    "does nothing",
]

TECHNICAL_ACTIONS = [
    "open the report",
    "click export",
    "sign in with SSO",
    "upload a CSV",
    "switch workspaces",
    "generate an invoice preview",
    "create a webhook",
    "save my changes",
]

FEATURE_REQUESTS = [
    "bulk approval workflows",
    "scheduled PDF exports",
    "role-based approval rules",
    "dark mode",
    "webhook retries with alerts",
    "SAML group mapping",
    "undo for deleted dashboards",
    "native Slack notifications",
]

FEATURE_SURFACES = [
    "admin console",
    "reporting module",
    "mobile app",
    "workflow builder",
    "API settings page",
    "audit log",
    "dashboard editor",
    "user management screen",
]

SERVICE_ISSUES = [
    "support has ignored three follow-ups for a week",
    "the last agent closed my case without solving anything",
    "your outage communication was vague and late",
    "the onboarding call was rushed and unhelpful",
    "the knowledge base article was misleading",
    "I keep getting transferred without any ownership",
    "the promised callback never happened",
    "the response I received was rude and dismissive",
]

OTHER_REQUESTS = [
    "a copy of your SOC 2 report",
    "the billing department's mailing address",
    "help updating our account owner email",
    "your W-9 form for vendor onboarding",
    "the date of the next training webinar",
    "pricing for a new region before we expand",
    "a sales demo for another business unit",
    "documentation on API rate limits",
]

CURATED_EVAL_EXAMPLES: list[tuple[str, str]] = [
    ("I was charged twice for the April invoice and still have not received a refund.", "billing"),
    ("Your invoice shows ten seats, but we only have six active users.", "billing"),
    ("The March bill still includes the analytics add-on that I removed last month.", "billing"),
    ("Why did my annual renewal bill use the old price instead of the contracted discount?", "billing"),
    ("I canceled on Friday, but my credit card was charged again this morning.", "billing"),
    ("The tax amount on invoice INV-2041 looks too high for our location.", "billing"),
    ("Please explain why the prorated credit is missing from my downgrade invoice.", "billing"),
    ("My receipt shows a currency conversion fee that was never mentioned before.", "billing"),
    ("We were billed for storage overages even though usage stayed below the limit.", "billing"),
    ("The invoice total does not match the quote your sales team sent us.", "billing"),
    ("I need a corrected invoice because the VAT number is wrong and the total is off.", "billing"),
    ("Our renewal should be monthly now, but the system billed us for a full year.", "billing"),
    ("The payment portal shows an overdue balance even after the wire cleared yesterday.", "billing"),
    ("I was charged once for the base plan and again for the exact same seats.", "billing"),
    ("Why did the invoice jump after we removed three contractors from the workspace?", "billing"),
    ("We were promised a service credit after the outage, but it is not on the invoice.", "billing"),
    ("The bill says we used premium support, but we never enabled that feature.", "billing"),
    ("I need help because my invoice keeps regenerating with the wrong subscription tier.", "billing"),
    ("The charge on my statement is higher than the amount shown inside the billing page.", "billing"),
    ("My cancellation was confirmed, yet the May subscription fee still posted to the card.", "billing"),
    ("The export to CSV button does nothing when I click it from the dashboard.", "technical_issue"),
    ("SSO login loops back to the sign-in page every time I try to access the admin console.", "technical_issue"),
    ("Uploading a file larger than 10 MB causes the web app to freeze indefinitely.", "technical_issue"),
    ("The mobile app crashes right after I tap the notifications tab.", "technical_issue"),
    ("Creating a webhook returns a 500 error even though the payload is valid JSON.", "technical_issue"),
    ("The page turns blank whenever I switch from one workspace to another.", "technical_issue"),
    ("Saving dashboard filters silently fails and my changes disappear after refresh.", "technical_issue"),
    ("I cannot generate the invoice preview because the screen just spins forever.", "technical_issue"),
    ("The API keeps timing out on a request that normally completes in a second.", "technical_issue"),
    ("Our report never finishes loading after I select the last 90 days.", "technical_issue"),
    ("The admin portal logs me out every time I open the user management screen.", "technical_issue"),
    ("Clicking save on the workflow builder has no effect and there is no error message.", "technical_issue"),
    ("The billing page shows a broken layout on Firefox and I cannot submit payment details.", "technical_issue"),
    ("We see duplicate webhook events because the retry status never updates.", "technical_issue"),
    ("The dashboard editor hangs when I add a second chart to the canvas.", "technical_issue"),
    ("My API token works in staging but every production call comes back unauthorized.", "technical_issue"),
    ("The mobile push notification opens the wrong report when tapped.", "technical_issue"),
    ("Each time I import the CSV template, the field mapping step resets itself.", "technical_issue"),
    ("The audit log page is stuck on a loading spinner for all admins in our tenant.", "technical_issue"),
    ("I get an unknown server error whenever I try to approve a pending request.", "technical_issue"),
    ("Please add scheduled PDF exports so our finance team does not have to run reports manually.", "feature_request"),
    ("It would be useful to have dark mode in the dashboard for evening shifts.", "feature_request"),
    ("We need SAML group mapping so access rules update automatically from our identity provider.", "feature_request"),
    ("Can you add webhook retry alerts to the admin console?", "feature_request"),
    ("Our approvers want bulk approval actions inside the workflow screen.", "feature_request"),
    ("Please support undo after deleting a dashboard by mistake.", "feature_request"),
    ("We would like native Slack notifications for failed jobs and escalations.", "feature_request"),
    ("Add role-based approval rules so only finance managers can approve large invoices.", "feature_request"),
    ("It would help if the reporting module could schedule a weekly email digest.", "feature_request"),
    ("Please add a read-only audit log export for compliance reviews.", "feature_request"),
    ("Our team wants a mobile approval flow instead of only desktop approvals.", "feature_request"),
    ("Can you support API usage quotas by workspace rather than only account-wide?", "feature_request"),
    ("I want dashboard widgets to snap into a grid automatically when dragged.", "feature_request"),
    ("Please add templated comments for approvers so recurring notes are faster to enter.", "feature_request"),
    ("We need the system to route requests based on cost center before final approval.", "feature_request"),
    ("Add a setting to mute low-priority notifications during weekends.", "feature_request"),
    ("It would be useful to compare two saved reports side by side.", "feature_request"),
    ("Can the product support multiple invoice approvers in sequence?", "feature_request"),
    ("Please add custom branding on exported PDFs for customer-facing reports.", "feature_request"),
    ("I would like a preview mode before publishing dashboard edits to the team.", "feature_request"),
    ("Support has ignored my last three emails and nobody owns the issue.", "complaint"),
    ("The last agent closed the case without solving anything and marked it resolved anyway.", "complaint"),
    ("Your outage update was vague, late, and honestly not acceptable.", "complaint"),
    ("The onboarding session felt rushed and we left with more questions than answers.", "complaint"),
    ("I keep getting transferred between teams instead of receiving a real answer.", "complaint"),
    ("The response from support was dismissive and did not address the actual problem.", "complaint"),
    ("Your documentation sent us in circles and wasted half a day.", "complaint"),
    ("I was promised a callback yesterday and nobody contacted me.", "complaint"),
    ("This is the second time a serious outage happened with almost no communication from your team.", "complaint"),
    ("The support portal is confusing and the ticket updates are impossible to follow.", "complaint"),
    ("My account manager keeps changing and no one seems accountable.", "complaint"),
    ("I am unhappy with how long it takes to get even a basic answer from support.", "complaint"),
    ("Your team keeps asking me to repeat the same details on every reply.", "complaint"),
    ("This experience has been frustrating because the issue was bounced around for days.", "complaint"),
    ("The way this case was handled was careless and unprofessional.", "complaint"),
    ("I need to file a complaint about the rude tone in your last message.", "complaint"),
    ("We lost confidence after the incident review blamed us without any evidence.", "complaint"),
    ("The promised fix date slipped again and nobody proactively informed us.", "complaint"),
    ("I do not want another canned response; I want someone to actually take ownership.", "complaint"),
    ("This has become a complaint because the service quality has dropped every month.", "complaint"),
    ("Can you send me your SOC 2 report for our vendor review?", "other"),
    ("What is the mailing address for your billing department?", "other"),
    ("I need to change the primary account owner email on our workspace.", "other"),
    ("Do you offer a training webinar for new administrators next month?", "other"),
    ("Please share your W-9 form so procurement can finish setup.", "other"),
    ("We are opening a new office and want pricing information for that region.", "other"),
    ("Can someone from sales contact our procurement team about a larger deployment?", "other"),
    ("Where can I find documentation on API rate limits and burst behavior?", "other"),
    ("Do you have a public status page we can subscribe to?", "other"),
    ("How do I update our company legal name on the account?", "other"),
    ("Can you confirm whether your support team is available on local holidays?", "other"),
    ("I need a copy of the signed order form for our records.", "other"),
    ("Which email should we use for security notifications from your side?", "other"),
    ("Please let me know if you support invoices in Japanese for our Tokyo office.", "other"),
    ("Do you provide implementation partners in the Middle East region?", "other"),
    ("We need the latest product brochure for an internal steering meeting.", "other"),
    ("Can you point me to your documentation for account deactivation steps?", "other"),
    ("I want to know the date of your next roadmap webinar.", "other"),
    ("Who should we contact to discuss a reseller partnership?", "other"),
    ("Please share the support escalation phone number for our records.", "other"),
]


def ensure_datasets() -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_PATH.exists():
        TRAIN_PATH.write_text(json.dumps(generate_training_examples(), indent=2), encoding="utf-8")
    if not EVAL_PATH.exists():
        EVAL_PATH.write_text(json.dumps(generate_eval_examples(), indent=2), encoding="utf-8")
    return TRAIN_PATH, EVAL_PATH


def generate_training_examples() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(_generate_for_label("billing", 200))
    rows.extend(_generate_for_label("technical_issue", 200))
    rows.extend(_generate_for_label("feature_request", 200))
    rows.extend(_generate_for_label("complaint", 200))
    rows.extend(_generate_for_label("other", 200))
    return rows


def generate_eval_examples() -> list[dict[str, str]]:
    return [{"text": text, "label": label} for text, label in CURATED_EVAL_EXAMPLES]


def load_train_examples() -> list[dict[str, str]]:
    ensure_datasets()
    return json.loads(TRAIN_PATH.read_text(encoding="utf-8"))


def load_eval_examples() -> list[dict[str, str]]:
    ensure_datasets()
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def _generate_for_label(label: str, count: int) -> list[dict[str, str]]:
    rng = random.Random(SEEDS[label])
    builders = {
        "billing": _billing_text,
        "technical_issue": _technical_text,
        "feature_request": _feature_text,
        "complaint": _complaint_text,
        "other": _other_text,
    }
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    while len(rows) < count:
        text = builders[label](rng)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": label})
    return rows


def _billing_text(rng: random.Random) -> str:
    template = rng.choice(BILLING_TEMPLATES)
    return template.format(
        opening=rng.choice(OPENINGS),
        issue=rng.choice(BILLING_ISSUES),
        product=rng.choice(BILLING_PRODUCTS),
        closing=rng.choice(CLOSINGS),
    )


def _technical_text(rng: random.Random) -> str:
    template = rng.choice(TECHNICAL_TEMPLATES)
    return template.format(
        opening=rng.choice(OPENINGS),
        surface=rng.choice(SURFACES),
        symptom=rng.choice(TECHNICAL_SYMPTOMS),
        action=rng.choice(TECHNICAL_ACTIONS),
        closing=rng.choice(CLOSINGS),
    )


def _feature_text(rng: random.Random) -> str:
    template = rng.choice(FEATURE_TEMPLATES)
    return template.format(
        opening=rng.choice(OPENINGS),
        request=rng.choice(FEATURE_REQUESTS),
        surface=rng.choice(FEATURE_SURFACES),
        closing=rng.choice(CLOSINGS),
    )


def _complaint_text(rng: random.Random) -> str:
    template = rng.choice(COMPLAINT_TEMPLATES)
    return template.format(
        opening=rng.choice(OPENINGS),
        service_issue=rng.choice(SERVICE_ISSUES),
        closing=rng.choice(CLOSINGS),
    )


def _other_text(rng: random.Random) -> str:
    template = rng.choice(OTHER_TEMPLATES)
    return template.format(
        opening=rng.choice(OPENINGS),
        request=rng.choice(OTHER_REQUESTS),
        closing=rng.choice(CLOSINGS),
    )
