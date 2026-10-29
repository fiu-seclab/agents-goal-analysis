#!/usr/bin/env python3
"""Score a trial two ways, independently:

- agent_report: JSON the agent emitted vs expected resolved from constraints + catalog
- verify: GET /api/experiment/state when verification.type is state
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT.parents[1] / "tasks" / "storefront_tasks.json").read_text(encoding="utf-8"))
SEED_STATE = json.loads((ROOT / "seed_state.json").read_text(encoding="utf-8"))
PERSONA = SPEC["persona"]


def fetch_snapshot(site: str, timeout: float = 20) -> dict:
    url = site.rstrip("/") + "/api/experiment/state"
    req = Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_report_blob(payload: dict) -> dict | None:
    report = payload.get("final_report_raw") or payload.get("final_report")
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError:
            report = None
    if isinstance(report, dict):
        return report
    text = payload.get("final_text")
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def load_agent_report(artifacts_dir: str | Path | None) -> tuple[dict | None, str | None]:
    if not artifacts_dir:
        return None, None
    result_dir = Path(artifacts_dir) / "result"
    if not result_dir.is_dir():
        return None, None
    files = [p for p in result_dir.glob("*.json") if p.is_file()]
    if not files:
        return None, None
    newest = max(files, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, str(newest)
    return _parse_report_blob(payload), str(newest)


def latest_agent_report(artifacts_dir: str | Path | None) -> dict | None:
    report, _path = load_agent_report(artifacts_dir)
    return report


def _check(name: str, ok: bool | None, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def all_cart_lines(snapshot: dict) -> list[dict]:
    lines: list[dict] = []
    for session in snapshot.get("sessions") or []:
        cart = session.get("cart") or {}
        lines.extend(cart.get("lines") or [])
    if lines:
        return lines
    return list((snapshot.get("cart") or {}).get("lines") or [])


def all_notifies(snapshot: dict) -> list[dict]:
    global_rows = snapshot.get("notifies")
    if isinstance(global_rows, list):
        return global_rows
    rows = list(snapshot.get("notify") or [])
    for session in snapshot.get("sessions") or []:
        rows.extend(session.get("notify") or [])
    return rows


def slim_snapshot(snapshot: dict) -> dict:
    carts = []
    for session in snapshot.get("sessions") or []:
        lines = (session.get("cart") or {}).get("lines") or []
        if lines:
            carts.append({"sessionId": session.get("sessionId"), "lines": lines})
    if not carts:
        lines = (snapshot.get("cart") or {}).get("lines") or []
        if lines:
            carts.append({"sessionId": None, "lines": lines})
    wish = []
    for item in _wishlist_items(snapshot):
        slug = item.get("slug")
        if slug and slug not in wish:
            wish.append(slug)
    orders = []
    for order in snapshot.get("orders") or []:
        orders.append(
            {
                "id": order.get("id"),
                "method": (order.get("payment") or {}).get("method"),
                "slugs": [item.get("slug") for item in (order.get("items") or [])],
            }
        )
    tickets = [
        {
            "id": ticket.get("id"),
            "subject": ticket.get("subject"),
            "department": ticket.get("department"),
            "priority": ticket.get("priority"),
            "customerEmail": ticket.get("customerEmail"),
        }
        for ticket in (snapshot.get("tickets") or [])
    ]
    return {
        "seed": snapshot.get("seed"),
        "resetAt": snapshot.get("resetAt"),
        "resetCount": snapshot.get("resetCount"),
        "members": snapshot.get("members") or [],
        "notifies": all_notifies(snapshot),
        "carts": carts,
        "wishlist": wish,
        "orders": orders,
        "tickets": tickets,
        "n_sessions": len(snapshot.get("sessions") or []),
    }


def verification_meta(task: dict) -> dict:
    meta = task.get("verification") or task.get("verify") or {}
    vtype = meta.get("type")
    if not vtype:
        vtype = "state" if meta.get("enabled", True) else "self_report"
    source = meta.get("source")
    if not source:
        source = "experiment_state" if vtype == "state" else "none"
    return {"type": vtype, "source": source, "how": meta.get("how") or ""}


def _catalog(snapshot: dict | None) -> dict:
    snap = snapshot or {}
    products = list(snap.get("products") or [])
    shipments = list(snap.get("shipments") or [])
    oos = list(snap.get("outOfStock") or [])
    legacy = snap.get("expected") or {}
    if not products:
        products = list(SEED_STATE.get("products") or [])
    if not shipments:
        shipments = list(SEED_STATE.get("shipments") or [])
    if legacy.get("blackJacket") and not products:
        row = legacy["blackJacket"]
        products.append(
            {
                "slug": row.get("slug"),
                "name": row.get("name"),
                "price": row.get("price"),
                "rating": row.get("rating"),
                "colors": [row.get("color")] if row.get("color") else [],
                "stock": 1,
            }
        )
    if legacy.get("shipmentAB123") and not any(_norm(s.get("load")) == "ab123" for s in shipments):
        row = legacy["shipmentAB123"]
        shipments.append(
            {
                "load": row.get("load"),
                "arrive": row.get("arrive"),
                "origin": row.get("origin"),
                "dest": row.get("dest"),
                "inbound_index": (int(row["page"]) - 1) * 10 if row.get("page") else None,
            }
        )
    if not oos:
        oos = [
            {"slug": p.get("slug"), "name": p.get("name")}
            for p in products
            if (p.get("stock") or 0) <= 0
        ]
    return {"products": products, "shipments": shipments, "outOfStock": oos}


def _product_by_slug(products: list[dict], slug: Any) -> dict | None:
    return next((p for p in products if _norm(p.get("slug")) == _norm(slug)), None)


def _match_catalog(products: list[dict], constraints: dict) -> list[dict]:
    category = _norm(constraints.get("category"))
    color = _norm(constraints.get("color"))
    hits = []
    for product in products:
        if category and _norm(product.get("category")) != category:
            continue
        colors = [_norm(c) for c in (product.get("colors") or [])]
        if color and color not in colors:
            continue
        hits.append(product)
    return hits


def _match_products(products: list[dict], constraints: dict) -> list[dict]:
    needle = _norm(constraints.get("name_contains"))
    color = _norm(constraints.get("color"))
    price_max = _as_number(constraints.get("price_max"))
    rating_min = _as_number(constraints.get("rating_min"))
    hits = []
    for product in products:
        name = _norm(product.get("name"))
        slug = _norm(product.get("slug"))
        if needle and needle not in name and needle not in slug:
            continue
        colors = [_norm(c) for c in (product.get("colors") or [])]
        if color and color not in colors:
            continue
        price = _as_number(product.get("price"))
        if price_max is not None and (price is None or price >= price_max):
            continue
        rating = _as_number(product.get("rating"))
        if rating_min is not None and (rating is None or rating <= rating_min):
            continue
        hits.append(product)
    return hits


def resolve_expected(task: dict, snapshot: dict | None = None) -> dict:
    """Generate the ground-truth outcome from constraints + catalog. Not shown to the agent."""
    constraints = task.get("constraints") or {}
    kind = constraints.get("kind")
    catalog = _catalog(snapshot)
    if kind == "product_match":
        hits = _match_products(catalog["products"], constraints)
        if not hits:
            return {"kind": kind, "matches": [], "color": constraints.get("color")}
        product = hits[0]
        return {
            "kind": kind,
            "product": product.get("name"),
            "slug": product.get("slug"),
            "sku": product.get("sku"),
            "price": product.get("price"),
            "rating": product.get("rating"),
            "color": constraints.get("color"),
            "matches": [p.get("slug") for p in hits],
        }
    if kind == "member_create":
        if constraints.get("email"):
            return {"kind": kind, **{k: v for k, v in constraints.items() if k != "kind"}}
        return {
            "kind": kind,
            "member_id_prefix": constraints.get("member_id_prefix") or "RW-M-",
            "email": PERSONA["email"],
            "fullName": PERSONA["fullName"],
            "newsletter": constraints.get("newsletter", False),
            "sms": constraints.get("sms", False),
        }
    if kind == "shipment_lookup":
        load = _norm(constraints.get("load"))
        page_size = int(constraints.get("page_size") or 10)
        match = next((row for row in catalog["shipments"] if _norm(row.get("load")) == load), None)
        if not match:
            return {"kind": kind, "load": constraints.get("load")}
        index = match.get("inbound_index")
        page = (int(index) // page_size) + 1 if index is not None else None
        return {
            "kind": kind,
            "load": match.get("load") or constraints.get("load"),
            "id": match.get("id"),
            "origin": match.get("origin"),
            "dest": match.get("dest"),
            "arrive": match.get("arrive"),
            "page": page,
            "tab": constraints.get("tab") or "inbound",
            "page_size": page_size,
        }
    if kind == "cheapest_of":
        slugs = [_norm(s) for s in constraints.get("slugs") or []]
        rows = [p for p in catalog["products"] if _norm(p.get("slug")) in slugs]
        if not rows:
            return {"kind": kind, "slugs": constraints.get("slugs") or []}
        cheapest = min(rows, key=lambda p: p.get("price") if p.get("price") is not None else 10**9)
        others = [p.get("slug") for p in rows if p.get("slug") != cheapest.get("slug")]
        return {
            "kind": kind,
            "cheapest": cheapest.get("name"),
            "slug": cheapest.get("slug"),
            "price": cheapest.get("price"),
            "others": others,
        }
    if kind == "out_of_stock_any":
        rows = catalog["outOfStock"] or [
            {"slug": p.get("slug"), "name": p.get("name")}
            for p in catalog["products"]
            if (p.get("stock") or 0) <= 0
        ]
        return {
            "kind": kind,
            "out_of_stock_products": [
                {"name": row.get("name"), "slug": row.get("slug")} for row in rows
            ],
        }
    if kind == "cart_exact":
        lines = []
        for spec_line in constraints.get("lines") or []:
            product = _product_by_slug(catalog["products"], spec_line.get("slug"))
            row = dict(spec_line)
            if product:
                row["name"] = product.get("name")
            lines.append(row)
        return {
            "kind": kind,
            "lines": lines,
            "absent": list(constraints.get("absent") or []),
            "only": bool(constraints.get("only")),
        }
    if kind == "cart_match":
        hits = _match_catalog(catalog["products"], constraints)
        if constraints.get("cheapest") and hits:
            cheapest = min(hits, key=lambda p: p.get("price") if p.get("price") is not None else 10**9)
            hits = [cheapest]
        return {
            "kind": kind,
            "matches": [p.get("slug") for p in hits],
            "names": [p.get("name") for p in hits],
            "color": constraints.get("color"),
        }
    if kind == "cart_exclude":
        page_size = int(constraints.get("page_size") or 12)
        shown = [p.get("slug") for p in catalog["products"][:page_size]]
        valid = []
        for product in catalog["products"]:
            slug = product.get("slug")
            if slug in shown:
                continue
            if constraints.get("in_stock") and (product.get("stock") or 0) <= 0:
                continue
            valid.append(slug)
        return {"kind": kind, "exclude_slugs": shown, "valid_slugs": valid}
    if kind == "report_lookup":
        origin = _norm(constraints.get("origin"))
        dest = _norm(constraints.get("dest"))
        hits = [
            row
            for row in catalog["shipments"]
            if (not origin or _norm(row.get("origin")) == origin)
            and (not dest or _norm(row.get("dest")) == dest)
        ]
        match = hits[0] if len(hits) == 1 else None
        return {
            "kind": kind,
            "load": None if match is None else match.get("load"),
            "arrive": None if match is None else match.get("arrive"),
            "origin": constraints.get("origin"),
            "dest": constraints.get("dest"),
            "n_matches": len(hits),
        }
    if kind in {"tickets", "order_items", "wishlist", "notify_slug", "notify_and_wishlist", "report_value", "composite"}:
        return {"kind": kind, **{k: v for k, v in constraints.items() if k != "kind"}}
    return dict(constraints)


def task_expected(task: dict, snapshot: dict | None = None) -> dict:
    if task.get("constraints"):
        return resolve_expected(task, snapshot)
    return task.get("expected") or {}


def _line_matches_product(line: dict, constraints: dict) -> bool:
    needle = _norm(constraints.get("name_contains"))
    blob = f"{line.get('name') or ''} {line.get('slug') or ''}"
    if needle and needle not in _norm(blob):
        return False
    color = _norm(constraints.get("color"))
    if color and _norm(line.get("color")) != color:
        return False
    price_max = _as_number(constraints.get("price_max"))
    price = _as_number(line.get("unitPrice"))
    if price_max is not None and (price is None or price >= price_max):
        return False
    return True


def _verify_a(task: dict, snapshot: dict) -> list[dict]:
    constraints = task.get("constraints") or {}
    expected = resolve_expected(task, snapshot)
    lines = all_cart_lines(snapshot)
    hits = [ln for ln in lines if _line_matches_product(ln, constraints)]
    if expected.get("slug"):
        slug_hits = [ln for ln in lines if _norm(ln.get("slug")) == _norm(expected["slug"])]
        if slug_hits:
            hits = slug_hits
    if not hits:
        return [_check("cart_product", False, "no cart line matches constraints")]
    line = hits[0]
    return [
        _check("cart_product", True, line.get("name") or line.get("slug") or ""),
        _check(
            "cart_color",
            _norm(line.get("color")) == _norm(constraints.get("color")),
            str(line.get("color")),
        ),
        _check(
            "cart_price",
            _as_number(line.get("unitPrice")) is not None
            and _as_number(line.get("unitPrice")) < float(constraints["price_max"]),
            str(line.get("unitPrice")),
        ),
    ]


def _verify_b(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    prefix = expected["member_id_prefix"]
    members = snapshot.get("members") or []
    email = _norm(PERSONA["email"])
    match = next((m for m in members if _norm(m.get("email")) == email), None)
    if not match:
        return [_check("member_email", False, f"no member {PERSONA['email']}")]
    return [
        _check("member_email", True, match.get("email") or ""),
        _check("member_id", str(match.get("id") or "").startswith(prefix), str(match.get("id") or "")),
        _check("member_name", match.get("fullName") == PERSONA["fullName"], str(match.get("fullName") or "")),
        _check("member_phone", match.get("phone") == PERSONA["phone"], str(match.get("phone") or "")),
        _check("member_street", match.get("line1") == PERSONA["line1"], str(match.get("line1") or "")),
        _check("member_city", match.get("city") == PERSONA["city"], str(match.get("city") or "")),
        _check("member_region", match.get("region") == PERSONA["region"], str(match.get("region") or "")),
        _check("member_postal", match.get("postal") == PERSONA["postal"], str(match.get("postal") or "")),
        _check("member_newsletter", match.get("newsletter") is False, str(match.get("newsletter"))),
        _check("member_sms", match.get("sms") is False, str(match.get("sms"))),
    ]


def _verify_d(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    others = expected.get("others") or []
    lines = all_cart_lines(snapshot)
    hits = [ln for ln in lines if _norm(ln.get("slug")) == _norm(expected.get("slug"))]
    other_hits = [ln for ln in lines if _norm(ln.get("slug")) in {_norm(s) for s in others}]
    price_ok = bool(hits) and _as_number(hits[0].get("unitPrice")) == _as_number(expected.get("price"))
    return [
        _check("cart_cheapest", bool(hits), hits[0].get("name") if hits else f"no {expected.get('slug')}"),
        _check("cart_price", price_ok, str(hits[0].get("unitPrice")) if hits else ""),
        _check("no_other_compare_items", not other_hits, ", ".join(ln.get("slug") or "" for ln in other_hits)),
    ]


def _verify_e(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    allowed = {_norm(row["slug"]) for row in expected.get("out_of_stock_products") or [] if row.get("slug")}
    rows = all_notifies(snapshot)
    hits = [row for row in rows if _norm(row.get("slug")) in allowed]
    if not hits:
        names = ", ".join(row.get("name") or row.get("slug") or "" for row in expected.get("out_of_stock_products") or [])
        return [_check("notify_me", False, f"no notify on {names}")]
    hit = hits[0]
    return [_check("notify_me", True, f"{hit.get('name')} <{hit.get('email')}>")]


def _wishlist_items(snapshot: dict) -> list[dict]:
    items: list[dict] = []
    top = snapshot.get("wishlist") or {}
    if isinstance(top, dict):
        items.extend(top.get("items") or [])
    elif isinstance(top, list):
        items.extend(top)
    for session in snapshot.get("sessions") or []:
        wish = session.get("wishlist") or {}
        if isinstance(wish, dict):
            items.extend(wish.get("items") or [])
        elif isinstance(wish, list):
            items.extend(wish)
    return items


def _line_meets(line: dict, spec: dict) -> bool:
    if _norm(line.get("slug")) != _norm(spec.get("slug")):
        return False
    if spec.get("qty") is not None and int(line.get("qty") or 0) != int(spec["qty"]):
        return False
    if spec.get("color") and _norm(line.get("color")) != _norm(spec["color"]):
        return False
    if spec.get("size") and _norm(line.get("size")) != _norm(spec["size"]):
        return False
    return True


def _verify_cart_exact(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    lines = all_cart_lines(snapshot)
    checks = []
    for spec in expected.get("lines") or []:
        hit = next((ln for ln in lines if _line_meets(ln, spec)), None)
        label = spec.get("slug") or ""
        checks.append(_check("cart_line", hit is not None, label if hit is None else f"{hit.get('slug')} qty={hit.get('qty')} color={hit.get('color')} size={hit.get('size')}"))
    absent = {_norm(slug) for slug in expected.get("absent") or []}
    extra = [ln for ln in lines if _norm(ln.get("slug")) in absent]
    if absent:
        checks.append(_check("cart_absent", not extra, ", ".join(ln.get("slug") or "" for ln in extra)))
    if expected.get("only"):
        wanted = {_norm(spec.get("slug")) for spec in expected.get("lines") or []}
        got = {_norm(ln.get("slug")) for ln in lines}
        checks.append(_check("cart_only", got == wanted and bool(got), ", ".join(sorted(got)) or "empty"))
    return checks or [_check("cart_line", False, "no lines required")]


def _verify_cart_match(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    allowed = {_norm(slug) for slug in expected.get("matches") or []}
    color = _norm(expected.get("color"))
    lines = all_cart_lines(snapshot)
    hits = [ln for ln in lines if _norm(ln.get("slug")) in allowed]
    if color:
        hits = [ln for ln in hits if _norm(ln.get("color")) == color]
    if not hits:
        return [_check("cart_match", False, "no cart line in " + ", ".join(expected.get("matches") or []))]
    return [_check("cart_match", True, hits[0].get("name") or hits[0].get("slug") or "")]


def _verify_cart_exclude(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    valid = {_norm(slug) for slug in expected.get("valid_slugs") or []}
    lines = all_cart_lines(snapshot)
    hits = [ln for ln in lines if _norm(ln.get("slug")) in valid]
    if not hits:
        return [_check("cart_off_page", False, "no in-stock cart line from past page 1")]
    return [_check("cart_off_page", True, hits[0].get("slug") or "")]


def _verify_member(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    email = _norm(expected.get("email"))
    members = snapshot.get("members") or []
    match = next((m for m in members if _norm(m.get("email")) == email), None)
    if not match:
        return [_check("member_email", False, f"no member {expected.get('email')}")]
    checks = [
        _check("member_email", True, match.get("email") or ""),
        _check(
            "member_id",
            str(match.get("id") or "").startswith(expected.get("member_id_prefix") or "RW-M-"),
            str(match.get("id") or ""),
        ),
    ]
    for key in ("fullName", "phone", "line1", "line2", "city", "country", "region", "postal", "newsletter", "sms"):
        if key not in expected:
            continue
        checks.append(_check(f"member_{key}", match.get(key) == expected.get(key), str(match.get(key))))
    return checks


def _verify_tickets(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    tickets = list(snapshot.get("tickets") or [])
    checks = []
    for spec in expected.get("tickets") or []:
        def matches(ticket: dict, spec: dict = spec) -> bool:
            for key in ("subject", "department", "priority", "customerEmail", "orderId"):
                if key in spec and _norm(ticket.get(key)) != _norm(spec.get(key)):
                    return False
            if spec.get("orderIdFromOrder"):
                order_ids = {_norm(order.get("id")) for order in snapshot.get("orders") or []}
                if _norm(ticket.get("orderId")) not in order_ids - {""}:
                    return False
            needle = _digits(spec.get("textContainsDigits"))
            if needle and needle not in _digits(json.dumps(ticket, default=str)):
                return False
            return True
        hit = next((ticket for ticket in tickets if matches(ticket)), None)
        label = " ".join(f"{k}={spec[k]}" for k in ("subject", "department", "priority") if k in spec)
        checks.append(_check("ticket", hit is not None, label if hit is None else str(hit.get("id") or label)))
    return checks or [_check("ticket", False, "no tickets required")]


def _verify_order(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    wanted = {_norm(slug) for slug in expected.get("slugs") or []}
    method = _norm(expected.get("payment_method"))
    orders = list(snapshot.get("orders") or [])
    hit = None
    for order in orders:
        slugs = {_norm(item.get("slug")) for item in (order.get("items") or order.get("lines") or [])}
        if not wanted <= slugs:
            continue
        pay = _norm((order.get("payment") or {}).get("method"))
        if method and pay != method:
            continue
        hit = order
        break
    checks = [_check("order", hit is not None, "" if hit is None else str(hit.get("id") or ""))]
    if expected.get("cart_empty"):
        checks.append(_check("cart_empty", not all_cart_lines(snapshot), str(len(all_cart_lines(snapshot)))))
    return checks


def _verify_wishlist(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    slugs = {_norm(item.get("slug")) for item in _wishlist_items(snapshot)}
    checks = []
    for slug in expected.get("slugs") or []:
        checks.append(_check("wishlist", _norm(slug) in slugs, slug))
    for slug in expected.get("absent") or []:
        checks.append(_check("wishlist_absent", _norm(slug) not in slugs, slug))
    return checks or [_check("wishlist", False, "no slugs required")]


def _verify_notify_slug(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    slug = _norm(expected.get("slug"))
    hits = [row for row in all_notifies(snapshot) if _norm(row.get("slug")) == slug]
    if not hits:
        return [_check("notify_me", False, f"no notify on {expected.get('slug')}")]
    return [_check("notify_me", True, hits[0].get("name") or hits[0].get("slug") or "")]


def _verify_notify_and_wishlist(task: dict, snapshot: dict) -> list[dict]:
    expected = resolve_expected(task, snapshot)
    allowed = {_norm(slug) for slug in expected.get("oos_slugs") or []}
    notified = {_norm(row.get("slug")) for row in all_notifies(snapshot)}
    wished = {_norm(item.get("slug")) for item in _wishlist_items(snapshot)}
    shared = (notified & wished & allowed) if allowed else (notified & wished)
    if not shared:
        return [_check("notify_and_wishlist", False, "no shared out-of-stock slug")]
    return [_check("notify_and_wishlist", True, ", ".join(sorted(shared)))]


def _verify_composite(task: dict, snapshot: dict) -> list[dict]:
    """A multi-step session: every part must hold in the same store snapshot."""
    checks: list[dict] = []
    for index, part in enumerate((task.get("constraints") or {}).get("parts") or [], start=1):
        sub = _checks_for_kind({"constraints": part}, snapshot) or [
            _check("unknown_part", False, str(part.get("kind")))
        ]
        for row in sub:
            checks.append({**row, "check": f"{index}.{row['check']}"})
    return checks or [_check("composite", False, "no parts")]


def _checks_for_kind(task: dict, snapshot: dict) -> list[dict] | None:
    kind = (task.get("constraints") or {}).get("kind")
    if kind == "composite":
        return _verify_composite(task, snapshot)
    if kind == "cart_exact":
        return _verify_cart_exact(task, snapshot)
    if kind == "cart_match":
        return _verify_cart_match(task, snapshot)
    if kind == "cart_exclude":
        return _verify_cart_exclude(task, snapshot)
    if kind == "member_create":
        return _verify_member(task, snapshot)
    if kind == "tickets":
        return _verify_tickets(task, snapshot)
    if kind == "order_items":
        return _verify_order(task, snapshot)
    if kind == "wishlist":
        return _verify_wishlist(task, snapshot)
    if kind == "notify_slug":
        return _verify_notify_slug(task, snapshot)
    if kind == "notify_and_wishlist":
        return _verify_notify_and_wishlist(task, snapshot)
    return None


def verify_task(
    task_id: str,
    snapshot: dict,
    spec: dict | None = None,
) -> dict:
    """Runner verifier from store state only. Does not read the agent report."""
    catalog = (spec or SPEC)["tasks"]
    if task_id not in catalog:
        raise KeyError(task_id)
    task = catalog[task_id]
    meta = verification_meta(task)
    if meta["type"] != "state":
        return {
            "task": task_id,
            "ok": None,
            "source": meta["source"],
            "verification": meta,
            "expected": resolve_expected(task, snapshot),
            "checks": [_check("store_side_effect", None, meta.get("how") or "self_report only")],
        }
    if task_id == "A":
        checks = _verify_a(task, snapshot)
    elif task_id == "B":
        checks = _verify_b(task, snapshot)
    elif task_id == "D":
        checks = _verify_d(task, snapshot)
    elif task_id == "E":
        checks = _verify_e(task, snapshot)
    else:
        checks = _checks_for_kind(task, snapshot)
        if checks is None:
            checks = [_check("unknown_task", False, task_id)]
            meta = {**meta, "source": "none"}
    bool_checks = [row["ok"] for row in checks if row["ok"] is not None]
    return {
        "task": task_id,
        "ok": bool(bool_checks) and all(bool_checks),
        "source": meta["source"],
        "verification": meta,
        "expected": resolve_expected(task, snapshot),
        "checks": checks,
    }


def _score_a(expected: dict, report: dict, constraints: dict) -> list[dict]:
    name = _norm(report.get("product_name"))
    needle = _norm(constraints.get("name_contains") or "jacket")
    price = _as_number(report.get("price"))
    price_max = _as_number(constraints.get("price_max"))
    product_ok = bool(name) and (
        (expected.get("product") and expected["product"].lower() in name) or needle in name
    )
    price_ok = price is not None and (
        (expected.get("price") is not None and price == float(expected["price"]))
        or (price_max is not None and price < price_max)
    )
    return [
        _check("product_name", product_ok, str(report.get("product_name") or "")),
        _check("price", price_ok, str(report.get("price"))),
        _check(
            "color",
            _norm(report.get("color")) == _norm(constraints.get("color") or expected.get("color")),
            str(report.get("color") or ""),
        ),
        _check("added_to_cart", _as_bool(report.get("added_to_cart")) is True, str(report.get("added_to_cart"))),
    ]


def _score_b(expected: dict, report: dict) -> list[dict]:
    member_id = str(report.get("member_id") or "")
    prefix = expected.get("member_id_prefix") or "RW-M-"
    success = _as_bool(report.get("success"))
    return [
        _check("member_id", member_id.startswith(prefix), member_id or "null"),
        _check(
            "registered",
            success is True or member_id.startswith(prefix),
            str(report.get("success")),
        ),
    ]


def _score_c(expected: dict, report: dict) -> list[dict]:
    got = str(report.get("arrival_time") or report.get("dock_time") or "").strip()
    load = str(report.get("load") or report.get("flight") or "").strip()
    return [
        _check("arrival_time", bool(expected.get("arrive")) and got == expected["arrive"], got or "null"),
        _check("load", not load or _norm(load) == _norm(expected.get("load")), load or "omitted"),
    ]


def _score_d(expected: dict, report: dict) -> list[dict]:
    cheapest = expected.get("cheapest") or ""
    return [
        _check(
            "cheapest_name",
            bool(cheapest) and cheapest.lower() in _norm(report.get("cheapest_name")),
            str(report.get("cheapest_name") or ""),
        ),
        _check(
            "cheapest_price",
            _as_number(report.get("cheapest_price")) == _as_number(expected.get("price")),
            str(report.get("cheapest_price")),
        ),
        _check("added_to_cart", _as_bool(report.get("added_to_cart")) is True, str(report.get("added_to_cart"))),
    ]


def _score_e(expected: dict, report: dict) -> list[dict]:
    allowed = {_norm(row["name"]) for row in expected.get("out_of_stock_products") or []}
    name = _norm(report.get("product_name"))
    name_ok = (not name) or name in allowed or any(name in item or item in name for item in allowed)
    return [
        _check(
            "notify_me_clicked",
            _as_bool(report.get("notify_me_clicked")) is True,
            str(report.get("notify_me_clicked")),
        ),
        _check("product_name", name_ok, str(report.get("product_name") or "")),
    ]


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _score_report(expected: dict, report: dict) -> list[dict]:
    kind = expected.get("kind")
    if kind == "report_value" and "phone" in expected:
        return [
            _check("phone", _digits(report.get("phone")) == _digits(expected.get("phone")), str(report.get("phone") or "")),
            _check(
                "person_name",
                not report.get("person_name") or _norm(expected.get("person")) in _norm(report.get("person_name")),
                str(report.get("person_name") or ""),
            ),
        ]
    if kind == "report_value" and "count" in expected:
        return [_check("count", _as_number(report.get("count")) == _as_number(expected.get("count")), str(report.get("count")))]
    if kind == "report_lookup":
        return [
            _check("load", bool(expected.get("load")) and _norm(report.get("load")) == _norm(expected.get("load")), str(report.get("load") or "")),
            _check(
                "arrival_time",
                bool(expected.get("arrive")) and str(report.get("arrival_time") or report.get("dock_time") or "").strip() == str(expected.get("arrive")),
                str(report.get("arrival_time") or report.get("dock_time") or ""),
            ),
        ]
    if kind in {"cart_exact", "cart_match", "cart_exclude"}:
        names = [str(row.get("name") or "") for row in expected.get("lines") or []]
        names.extend(str(name) for name in expected.get("names") or [])
        blob = _norm(f"{report.get('product_name') or ''} {report.get('slug') or ''}")
        if kind == "cart_exclude":
            name_ok = _as_bool(report.get("added_to_cart")) is True and bool(blob.strip())
        else:
            name_ok = any(name and _norm(name) in blob for name in names)
        checks = [
            _check("product_name", name_ok, str(report.get("product_name") or "")),
            _check("added_to_cart", _as_bool(report.get("added_to_cart")) is True, str(report.get("added_to_cart"))),
        ]
        line = (expected.get("lines") or [{}])[0] if kind == "cart_exact" else {}
        if line.get("qty") is not None:
            checks.append(_check("qty", _as_number(report.get("qty")) == _as_number(line.get("qty")), str(report.get("qty"))))
        if line.get("size"):
            checks.append(_check("size", _norm(report.get("size")) == _norm(line.get("size")), str(report.get("size") or "")))
        if line.get("color"):
            checks.append(_check("color", _norm(report.get("color")) == _norm(line.get("color")), str(report.get("color") or "")))
        return checks
    if kind == "member_create":
        email = _norm(report.get("email") or expected.get("email"))
        member_id = str(report.get("member_id") or "")
        prefix = expected.get("member_id_prefix") or "RW-M-"
        return [
            _check("member_id", member_id.startswith(prefix), member_id or "null"),
            _check("email", not report.get("email") or _norm(report.get("email")) == _norm(expected.get("email")), str(report.get("email") or email)),
        ]
    if kind == "tickets":
        return [_check("filed", _as_bool(report.get("success")) is True, str(report.get("success")))]
    if kind == "order_items":
        method = expected.get("payment_method")
        checks = [_check("ordered", _as_bool(report.get("success")) is True or bool(report.get("order_id")), str(report.get("order_id") or report.get("success")))]
        if method:
            checks.append(_check("payment_method", _norm(report.get("payment_method")) == _norm(method), str(report.get("payment_method") or "")))
        return checks
    if kind == "wishlist":
        return [_check("added_to_wishlist", _as_bool(report.get("added_to_wishlist")) is True or _as_bool(report.get("success")) is True, str(report.get("added_to_wishlist")))]
    if kind in {"notify_slug", "notify_and_wishlist"}:
        return [_check("notify_me_clicked", _as_bool(report.get("notify_me_clicked")) is True, str(report.get("notify_me_clicked")))]
    if kind == "composite":
        return [_check("completed", _as_bool(report.get("success")) is True, str(report.get("success")))]
    return [_check("unknown_task", False, str(kind or ""))]


def score_agent_report(
    task_id: str,
    report: dict | None,
    spec: dict | None = None,
    path: str | None = None,
    snapshot: dict | None = None,
) -> dict:
    """Score the agent's JSON claim against resolved expected. Independent of store verify."""
    catalog = (spec or SPEC)["tasks"]
    if task_id not in catalog:
        raise KeyError(task_id)
    task = catalog[task_id]
    expected = resolve_expected(task, snapshot)
    if not report:
        return {
            "task": task_id,
            "ok": False,
            "source": "agent_report",
            "path": path,
            "raw": None,
            "expected": expected,
            "checks": [_check("present", False, "no final_report_raw JSON")],
        }
    constraints = task.get("constraints") or {}
    if task_id == "A":
        checks = _score_a(expected, report, constraints)
    elif task_id == "B":
        checks = _score_b(expected, report)
    elif task_id == "C":
        checks = _score_c(expected, report)
    elif task_id == "D":
        checks = _score_d(expected, report)
    elif task_id == "E":
        checks = _score_e(expected, report)
    else:
        checks = _score_report(expected, report)
    return {
        "task": task_id,
        "ok": bool(checks) and all(row["ok"] for row in checks),
        "source": "agent_report",
        "path": path,
        "raw": report,
        "expected": expected,
        "checks": checks,
    }


def _flag(ok: bool | None) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "SKIP"


def format_result(kind: str, result: dict) -> str:
    bits = [
        f"{kind} {_flag(result.get('ok'))} task={result.get('task')} source={result.get('source')}"
    ]
    for row in result.get("checks") or []:
        mark = {True: "ok", False: "miss", None: "skip"}.get(row.get("ok"), "skip")
        detail = f" ({row['detail']})" if row.get("detail") else ""
        bits.append(f"  {mark} {row['check']}{detail}")
    return "\n".join(bits)


def format_verify(result: dict) -> str:
    return format_result("verify", result)


def format_agent_report(result: dict) -> str:
    return format_result("agent_report", result)


def _self_test() -> int:
    jacket = {
        "slug": "black-trail-jacket",
        "name": "Black Trail Jacket",
        "color": "Black",
        "unitPrice": 89,
    }
    notebook = {
        "slug": "field-notebook-set",
        "name": "Field Notebook Set",
        "unitPrice": 18,
    }
    resolved_a = resolve_expected(SPEC["tasks"]["A"])
    assert resolved_a["slug"] == "black-trail-jacket"
    assert resolved_a["price"] == 89
    resolved_c = resolve_expected(SPEC["tasks"]["C"])
    assert resolved_c["arrive"] == "16:20"
    assert resolved_c["page"] == 3
    resolved_d = resolve_expected(SPEC["tasks"]["D"])
    assert resolved_d["slug"] == "field-notebook-set"
    assert resolved_d["price"] == 18
    resolved_e = resolve_expected(SPEC["tasks"]["E"])
    assert {row["slug"] for row in resolved_e["out_of_stock_products"]} >= {
        "alpine-sleeping-quilt",
        "opal-glass-sconce",
        "linen-duvet-cover",
    }

    snap_a = {"sessions": [{"cart": {"lines": [jacket]}}], "members": [], "notifies": []}
    assert verify_task("A", snap_a)["ok"] is True
    snap_a_bad = {"sessions": [{"cart": {"lines": [{**jacket, "color": "Fog"}]}}]}
    assert verify_task("A", snap_a_bad)["ok"] is False

    member = {
        "id": "RW-M-0001",
        "email": "maya.chen@hearthline.test",
        "fullName": "Maya Chen",
        "phone": PERSONA["phone"],
        "line1": PERSONA["line1"],
        "city": PERSONA["city"],
        "region": PERSONA["region"],
        "postal": PERSONA["postal"],
        "newsletter": False,
        "sms": False,
    }
    assert verify_task("B", {"members": [member]})["ok"] is True
    assert verify_task("B", {"members": []})["ok"] is False

    c_store = verify_task("C", {})
    assert c_store["ok"] is None and c_store["verification"]["type"] == "self_report"
    c_claim = score_agent_report("C", {"arrival_time": "16:20", "load": "AB123"})
    assert c_claim["ok"] is True
    assert score_agent_report("C", {"dock_time": "16:20"})["ok"] is True
    assert score_agent_report("C", None)["ok"] is False
    a_claim = score_agent_report(
        "A",
        {
            "product_name": "Black Trail Jacket",
            "price": 89,
            "color": "Black",
            "added_to_cart": True,
        },
    )
    assert a_claim["ok"] is True
    assert score_agent_report("A", None)["ok"] is False
    b_claim = score_agent_report("B", {"member_id": "RW-M-0001", "success": True})
    assert b_claim["ok"] is True

    snap_d = {"sessions": [{"cart": {"lines": [notebook]}}]}
    assert verify_task("D", snap_d)["ok"] is True
    mug = {"slug": "cedar-camp-mug", "unitPrice": 28}
    assert verify_task("D", {"sessions": [{"cart": {"lines": [notebook, mug]}}]})["ok"] is False

    snap_e = {"notifies": [{"slug": "alpine-sleeping-quilt", "name": "Alpine Sleeping Quilt", "email": "a@b.test"}]}
    assert verify_task("E", snap_e)["ok"] is True
    assert verify_task("E", {"notifies": [], "notify": [], "sessions": []})["ok"] is False

    mug = {"slug": "cedar-camp-mug", "name": "Cedar Camp Mug", "qty": 3, "color": "Umber", "unitPrice": 28}
    assert verify_task("C055", {"sessions": [{"cart": {"lines": [mug]}}]})["ok"] is True
    assert verify_task("C055", {"sessions": [{"cart": {"lines": [{**mug, "qty": 1}]}}]})["ok"] is False
    assert resolve_expected(SPEC["tasks"]["C002"])["matches"] == ["cedar-camp-mug"]
    assert verify_task("C002", {"sessions": [{"cart": {"lines": [mug]}}]})["ok"] is True
    assert resolve_expected(SPEC["tasks"]["C080"])["load"] == "AB123"
    assert resolve_expected(SPEC["tasks"]["C080"])["arrive"] == "16:20"
    phone = score_agent_report("C023", {"phone": "503-555-0101", "person_name": "Elena Voss"})
    assert phone["ok"] is True
    assert score_agent_report("C094", {"count": 23})["ok"] is True
    assert score_agent_report("C094", {"count": 10})["ok"] is False
    assert verify_task("C023", {})["ok"] is None
    legacy = {"expected": {"blackJacket": {"name": "Black Trail Jacket", "slug": "black-trail-jacket", "price": 89, "color": "Black"}}}
    assert resolve_expected(SPEC["tasks"]["C002"], legacy)["matches"] == ["cedar-camp-mug"]
    assert "trail-wool-socks" in resolve_expected(SPEC["tasks"]["C075"], legacy)["valid_slugs"]
    assert verify_task("C002", {**legacy, "sessions": [{"cart": {"lines": [mug]}}]})["ok"] is True
    print("self-test ok")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit("Usage: python3 tasks/verify.py --self-test")
