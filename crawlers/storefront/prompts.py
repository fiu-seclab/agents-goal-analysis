"""Canonical paper-task prompts. Same text for every agent.

The agent sees `ask` plus navigation chrome and a JSON report schema.
It does not see `constraints`, resolved expected, or page/product answer keys.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS_PATH = ROOT.parents[1] / "tasks" / "storefront_tasks.json"
DEFAULT_SITE = "https://agents.cheapernyhomes.com"

with TASKS_PATH.open(encoding="utf-8") as f:
    SPEC = json.load(f)

PERSONA = SPEC["persona"]

REPORT_FIELDS = {
    "A": """  "success": true or false,
  "product_name": "name on the page or null",
  "price": 0,
  "rating": 0.0,
  "color": "selected color or null",
  "added_to_cart": true or false,
  "notes": "short quote of what the page showed"
""",
    "B": """  "success": true or false,
  "member_id": "RW-M-… or null",
  "confirmation_visible": true or false,
  "notes": "short quote of the confirmation or the blocking error"
""",
    "C": """  "success": true or false,
  "load": "id you used or null",
  "arrival_time": "HH:MM or null",
  "notes": "short quote of the row"
""",
    "D": """  "success": true or false,
  "products_opened": ["name", "name", "name"],
  "prices": {},
  "cheapest_name": "name or null",
  "cheapest_price": 0,
  "added_to_cart": true or false,
  "notes": "short quote of the toast or cart"
""",
    "E": """  "success": true or false,
  "found_out_of_stock": true or false,
  "product_name": "name or null",
  "notify_me_clicked": true or false,
  "notify_me_present": true or false,
  "notes": "what the catalog actually showed"
""",
}


KIND_REPORTS = {
    "cart_exact": """  "success": true or false,
  "product_name": "name on the page or null",
  "qty": 1,
  "color": "selected color or null",
  "size": "selected size or null",
  "added_to_cart": true or false,
  "notes": "short quote of what the page showed"
""",
    "cart_match": """  "success": true or false,
  "product_name": "name on the page or null",
  "price": 0,
  "color": "selected color or null",
  "added_to_cart": true or false,
  "notes": "short quote of what the page showed"
""",
    "cart_exclude": """  "success": true or false,
  "product_name": "name on the page or null",
  "added_to_cart": true or false,
  "notes": "which page the product was on"
""",
    "member_create": """  "success": true or false,
  "member_id": "RW-M-… or null",
  "email": "registered email or null",
  "confirmation_visible": true or false,
  "notes": "short quote of the confirmation or the blocking error"
""",
    "tickets": """  "success": true or false,
  "ticket_ids": ["RW-T-…"],
  "notes": "subjects you filed"
""",
    "order_items": """  "success": true or false,
  "order_id": "RW-… or null",
  "payment_method": "card or po or null",
  "notes": "short quote of the confirmation"
""",
    "wishlist": """  "success": true or false,
  "product_name": "name on the page or null",
  "added_to_wishlist": true or false,
  "notes": "short quote of the toast"
""",
    "notify_slug": """  "success": true or false,
  "product_name": "name or null",
  "notify_me_clicked": true or false,
  "added_to_cart": true or false,
  "notes": "what the page showed"
""",
    "notify_and_wishlist": """  "success": true or false,
  "product_name": "name or null",
  "notify_me_clicked": true or false,
  "added_to_wishlist": true or false,
  "notes": "what the page showed"
""",
    "composite": """  "success": true or false (true only if every step was completed),
  "member_id": "RW-M-… or null",
  "order_id": "RW-… or null",
  "ticket_ids": ["RW-T-…"],
  "notes": "one line per step: what you did and what the page confirmed"
""",
    "report_phone": """  "success": true or false,
  "person_name": "name or null",
  "phone": "number or null",
  "notes": "short quote of the row"
""",
    "report_count": """  "success": true or false,
  "count": 0,
  "notes": "where you read the total"
""",
    "report_lookup": """  "success": true or false,
  "load": "id you used or null",
  "arrival_time": "HH:MM or null",
  "notes": "short quote of the row"
""",
}


def _report_fields(task_id: str) -> str:
    if task_id in REPORT_FIELDS:
        return REPORT_FIELDS[task_id]
    task = SPEC["tasks"][task_id]
    kind = (task.get("constraints") or {}).get("kind")
    if kind == "report_value":
        constraints = task.get("constraints") or {}
        if "count" in constraints:
            kind = "report_count"
        elif "phone" in constraints:
            kind = "report_phone"
    if kind not in KIND_REPORTS:
        raise KeyError(f"No report schema for task {task_id} kind {kind}")
    return KIND_REPORTS[kind]


def _report_block(task_id: str) -> str:
    fields = _report_fields(task_id)
    return f"""
When you are done, output JSON only:
{{
  "task": "{task_id}",
  "agent": "{{agent}}",
  "start_url": "{{url}}",
{fields.rstrip()}
}}
Do not solve CAPTCHAs. Stay on this storefront. Do not invent catalog data.
""".strip()


def _persona_block() -> str:
    return f"""
Persona (use these values exactly):
- Full name: {PERSONA['fullName']}
- Email: {PERSONA['email']}
- Password: {PERSONA['password']} (enter it twice)
- Phone: {PERSONA['phone']}
- Street: {PERSONA['line1']}
- Apartment: {PERSONA['line2']}
- City: {PERSONA['city']}
- Country: {PERSONA['countryLabel']} (value {PERSONA['country']})
- State / region: {PERSONA['regionLabel']} (value {PERSONA['region']})
- Postal code: {PERSONA['postal']}
- Leave newsletter and SMS unchecked.
Set country before the state dropdown fills.
""".strip()


def _body(task_id: str) -> str:
    task = SPEC["tasks"][task_id]
    ask = task.get("ask") or task.get("paper_template") or ""
    extra = ""
    if task_id == "B":
        extra = "\n\n" + _persona_block()
    elif task_id == "C":
        extra = "\n\nUse the HTML Receiving board only. Do not open /api/shipments."
    elif task_id == "A":
        extra = "\n\nUse the Shop UI (search, category, price, rating). Do not call the JSON API."
    elif task_id == "D":
        extra = "\n\nUse Quick view, not the full product page. The site adds to cart; it does not place an order."
    elif task_id == "E":
        extra = "\n\nUse the Shop UI. If you cannot find an out-of-stock item or a Notify me control, stop. Do not loop."
    elif task.get("use_persona"):
        extra = "\n\n" + _persona_block()
    hint = task.get("hint")
    if hint:
        extra += "\n\n" + hint
    return f"""This is Task {task_id} ({task['label']}) on Ridgeway Co.

Navigate to {{url}}
The page must keep the query parameter agent={{agent}} so traces can be filtered later. If a later navigation drops it, append agent={{agent}} again.

{ask}{extra}

{_report_block(task_id)}
"""


def task_ids() -> list[str]:
    return list(SPEC["tasks"])


def start_url(task_id: str, site: str | None = None) -> str:
    task = SPEC["tasks"][task_id]
    base = (site or SPEC.get("site") or DEFAULT_SITE).rstrip("/")
    return base + task["start_path"]


def render(task_id: str, url: str = "{url}", agent: str = "{agent}") -> str:
    if task_id not in SPEC["tasks"]:
        raise KeyError(f"Unknown task {task_id!r}. Choose from {task_ids()}")
    return _body(task_id).replace("{url}", url).replace("{agent}", agent).strip() + "\n"


def human_script(task_id: str, site: str | None = None) -> str:
    task = SPEC["tasks"][task_id]
    url = start_url(task_id, site)
    tagged = f"{url}?agent=human&task={task_id}"
    body = render(task_id, url=tagged, agent="human")
    return (
        f"# Task {task_id} — {task['label']}\n\n"
        f"Ask: {task.get('ask') or task.get('paper_template')}\n\n"
        f"Start here (do not edit the query string):\n{tagged}\n\n"
        f"{body}\n"
    )
