#!/usr/bin/env python3
"""Role-based action tokens for causal models.

One token per action on the action clock (`effects_exp.action_spans`): pointer
motion and focus fold into the action they lead to, key runs, click chords and
scroll bursts collapse to one, consecutive page loads collapse to one. Raw
pointer volume therefore does not leak the humanization setting.

A token says what kind of action it was and what *role* of control it hit
(text box, password box, link, button, ...). It never says which control: no
element id, field name, url, path, agent, or task enters a feature. Whether the
page changed or was seen before is kept as a bit; the page itself is not.

Everything here is causal: token k depends only on events up to its own end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from effects_exp import action_spans, kind_of  # noqa: E402

KINDS = ["type", "click", "scroll", "nav", "change", "submit", "select", "reset"]
ROLES = ["none", "text", "password", "choice", "select", "textarea", "button", "link", "form", "content"]
KIND_INDEX = {name: i for i, name in enumerate(KINDS)}
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}
N_KINDS = len(KINDS)
N_ROLES = len(ROLES)
N_TOKENS = N_KINDS * N_ROLES
NUM_NAMES = ["gap", "burst", "keys", "elapsed", "new_page", "revisit"]
N_NUM = len(NUM_NAMES)
KEYDOWN = "keydown"
MIN_ACTIONS = 1


def role_of(event: dict) -> str:
    tag = (event.get("target_tag") or "").lower()
    kind = (event.get("target_type") or "").lower()
    if not tag:
        return "none"
    if tag == "input":
        if kind == "password":
            return "password"
        if kind in {"checkbox", "radio"}:
            return "choice"
        if kind in {"submit", "button", "image", "reset"}:
            return "button"
        return "text"
    return {
        "textarea": "textarea",
        "select": "select",
        "button": "button",
        "a": "link",
        "form": "form",
    }.get(tag, "content")


@dataclass
class Tokens:
    ids: np.ndarray        # (n,) int, kind * N_ROLES + role
    num: np.ndarray        # (n, N_NUM) float, scaled to roughly [0, 1]
    t_ms: np.ndarray       # (n,) elapsed ms at the end of each action
    pages: np.ndarray      # (n,) distinct pages seen up to and including this action
    ev_end: np.ndarray     # (n,) index in the raw stream of the event ending each action

    def __len__(self) -> int:
        return int(self.ids.shape[0])


LOGIN_FIRST_PASSWORD_BY = 12  # a sign-in opens the trace if a password box appears this early


def login_cut(tok: Tokens) -> int | None:
    """Actions that belong to the opening sign-in, or None if the trace has none.

    The collection prompts told the agent to sign in first, so the opening of
    every ServiceNow and Gitea trace is a login, not the task. The sign-in ends
    at the first page load after the first password box. Do not use this where
    a password box is part of the task from the start (account creation).
    """
    role = tok.ids % N_ROLES
    boxes = np.where(role == ROLE_INDEX["password"])[0]
    if len(boxes) == 0 or boxes[0] > LOGIN_FIRST_PASSWORD_BY:
        return None
    kind = tok.ids // N_ROLES
    for i in range(int(boxes[0]), len(tok)):
        if kind[i] == KIND_INDEX["nav"]:
            return i + 1
    return len(tok)


def tokenize(events: list[dict], warp: float = 1.0, trim_login: bool = False) -> Tokens | None:
    """Turn a raw event stream into action tokens. `warp` stretches time.

    With `trim_login` the opening sign-in is cut and the clock restarts after it.
    """
    tok = _tokenize(events, warp)
    if tok is None or not trim_login:
        return tok
    cut = login_cut(tok)
    if cut is None:
        return tok
    if cut >= len(tok):
        return None
    return _tokenize(events[int(tok.ev_end[cut - 1]) + 1 :], warp)


def _tokenize(events: list[dict], warp: float) -> Tokens | None:
    ids: list[int] = []
    num: list[list[float]] = []
    times: list[float] = []
    pages: list[int] = []
    ends: list[int] = []
    t0 = float(events[0].get("t_rel_ms") or 0) if events else 0.0
    seen: set[str] = set()
    prev_path: str | None = None
    prev_t = 0.0
    last_kind = None
    for lead, end in action_spans(events):
        event = events[end]
        name = event.get("event_name") or ""
        kind = kind_of(name)
        if kind == "other":
            if name in {"select", "reset"}:
                kind = name
            else:
                continue
        if kind is None:
            continue
        if kind == "nav" and last_kind == "nav":
            continue
        first = lead
        while first < end and kind_of(events[first].get("event_name") or "") is None:
            first += 1
        t_end = (float(event.get("t_rel_ms") or 0) - t0) * warp
        t_first = (float(events[first].get("t_rel_ms") or 0) - t0) * warp
        keys = sum(1 for row in events[first : end + 1] if row.get("event_name") == KEYDOWN)
        path = event.get("path") or ""
        new_page = prev_path is not None and path != prev_path
        revisit = new_page and path in seen
        seen.add(path)
        prev_path = path
        role = "none" if kind == "nav" else role_of(event)
        ids.append(KIND_INDEX[kind] * N_ROLES + ROLE_INDEX[role])
        num.append([
            np.log1p(max(0.0, t_first - prev_t)) / 10.0,
            np.log1p(max(0.0, t_end - t_first)) / 10.0,
            np.log1p(keys) / 4.0,
            np.log1p(t_end) / 12.0,
            1.0 if new_page else 0.0,
            1.0 if revisit else 0.0,
        ])
        times.append(t_end)
        pages.append(len(seen))
        ends.append(end)
        prev_t = t_end
        last_kind = kind
    if len(ids) < MIN_ACTIONS:
        return None
    return Tokens(
        ids=np.asarray(ids, dtype=np.int64),
        num=np.asarray(num, dtype=np.float32),
        t_ms=np.asarray(times, dtype=np.float64),
        pages=np.asarray(pages, dtype=np.int64),
        ev_end=np.asarray(ends, dtype=np.int64),
    )


# ----------------------------------------------------------------------------
# Forest features at every prefix length, built from cumulative sums.
# ----------------------------------------------------------------------------

RECENT = 8
BASE_NAMES: list[str] = (
    [f"frac:{KINDS[i // N_ROLES]}/{ROLES[i % N_ROLES]}" for i in range(N_TOKENS)]
    + [f"recent:{kind}" for kind in KINDS]
    + [f"last_kind:{kind}" for kind in KINDS]
    + [f"last_role:{role}" for role in ROLES]
    + [
        "log_actions",
        "log_elapsed_s",
        "log_keys",
        "keys_per_type",
        "mean_gap",
        "mean_burst",
        "log_pages",
        "new_page_rate",
        "revisit_rate",
        "n_submit_log",
        "n_change_log",
        "any_password",
    ]
)


def prefix_features(tok: Tokens) -> np.ndarray:
    """Row k-1 holds the features of the first k actions."""
    n = len(tok)
    kind = tok.ids // N_ROLES
    role = tok.ids % N_ROLES
    pair = np.zeros((n, N_TOKENS), dtype=np.float64)
    pair[np.arange(n), tok.ids] = 1.0
    kinds = np.zeros((n, N_KINDS), dtype=np.float64)
    kinds[np.arange(n), kind] = 1.0
    roles = np.zeros((n, N_ROLES), dtype=np.float64)
    roles[np.arange(n), role] = 1.0
    count = np.arange(1, n + 1, dtype=np.float64)[:, None]
    cum_pair = np.cumsum(pair, axis=0)
    cum_kinds = np.cumsum(kinds, axis=0)
    shifted = np.vstack([np.zeros((RECENT, N_KINDS)), cum_kinds])[:n]
    window = np.minimum(count, RECENT)
    recent = (cum_kinds - shifted) / window
    keys = np.expm1(tok.num[:, 2] * 4.0)
    cum_keys = np.cumsum(keys)
    n_type = np.maximum(1.0, cum_kinds[:, KIND_INDEX["type"]])
    cum = lambda col: np.cumsum(tok.num[:, col]) / count[:, 0]  # noqa: E731
    extra = np.column_stack([
        np.log1p(count[:, 0]),
        np.log1p(tok.t_ms / 1000.0),
        np.log1p(cum_keys),
        np.log1p(cum_keys / n_type),
        cum(0),
        cum(1),
        np.log1p(tok.pages.astype(np.float64)),
        cum(4),
        cum(5),
        np.log1p(cum_kinds[:, KIND_INDEX["submit"]]),
        np.log1p(cum_kinds[:, KIND_INDEX["change"]]),
        (np.cumsum(role == ROLE_INDEX["password"]) > 0).astype(np.float64),
    ])
    return np.hstack([cum_pair / count, recent, kinds, roles, extra])


def bigram_features(tok: Tokens, vocab: dict[tuple[int, int], int]) -> np.ndarray:
    """Cumulative bigram frequencies over (token, next token), for the n-gram control."""
    n = len(tok)
    grams = np.zeros((n, len(vocab)), dtype=np.float64)
    for i in range(1, n):
        col = vocab.get((int(tok.ids[i - 1]), int(tok.ids[i])))
        if col is not None:
            grams[i, col] = 1.0
    return np.cumsum(grams, axis=0) / np.arange(1, n + 1, dtype=np.float64)[:, None]


def bigram_vocab(sessions: list[Tokens], min_sessions: int = 3) -> dict[tuple[int, int], int]:
    seen: dict[tuple[int, int], int] = {}
    for tok in sessions:
        for pair in {(int(a), int(b)) for a, b in zip(tok.ids[:-1], tok.ids[1:])}:
            seen[pair] = seen.get(pair, 0) + 1
    keep = sorted(pair for pair, c in seen.items() if c >= min_sessions)
    return {pair: i for i, pair in enumerate(keep)}


# ----------------------------------------------------------------------------
# Stage 2 schema-grounded features (causal per action prefix, UUIDs stripped).
# ----------------------------------------------------------------------------

import re
from collections import Counter

HEX_OR_NUM = re.compile(r"^[0-9a-f]{8,}$|^\d+$")


def _post_login_events(events: list[dict], trim_login: bool) -> list[dict]:
    if not trim_login:
        return events
    full = _tokenize(events, 1.0)
    if full is None:
        return events
    cut = login_cut(full)
    if cut is None:
        return events
    if cut >= len(full):
        return []
    return events[int(full.ev_end[cut - 1]) + 1 :]


def action_site_dicts(events: list[dict], tok: Tokens, trim_login: bool = False) -> list[Counter]:
    """Return a list of length len(tok), where entry a holds the schema tokens
    observed during action span a (causal: only events up to tok.ev_end[a]).
    """
    tail = _post_login_events(events, trim_login)
    n = len(tok)
    per_action: list[Counter] = []
    prev_end = -1
    for a in range(n):
        end = int(tok.ev_end[a])
        span_evs = tail[prev_end + 1 : end + 1]
        prev_end = end
        c = Counter()
        for e in span_evs:
            tag = (e.get("target_tag") or "").lower().strip()
            tid = (e.get("target_id") or "").lower().strip()
            tname = (e.get("target_name") or "").lower().strip()
            ttype = (e.get("target_type") or "").lower().strip()
            path = (e.get("path") or "").lower().strip()
            url = (e.get("url") or "").lower().strip()
            el = (e.get("element") or "").lower().strip()

            if tag:
                c[f"tag:{tag}"] += 1
            if ttype and not HEX_OR_NUM.search(ttype):
                c[f"type:{ttype}"] += 1
            for raw, pfx in ((tid, "idtok"), (tname, "nametok")):
                if raw and not HEX_OR_NUM.search(raw):
                    c[f"raw_{pfx}:{raw}"] += 1
                    for part in re.split(r"[^a-z0-9]+", raw):
                        if len(part) >= 2 and not HEX_OR_NUM.search(part):
                            c[f"{pfx}:{part}"] += 1
            if path:
                for seg in re.split(r"[^a-z0-9]+", path):
                    if len(seg) >= 2 and not HEX_OR_NUM.search(seg):
                        c[f"pseg:{seg}"] += 1
            if "?" in url:
                q = url.split("?", 1)[1]
                for part in re.split(r"[^a-z0-9]+", q):
                    if len(part) >= 2 and not HEX_OR_NUM.search(part):
                        c[f"qtok:{part}"] += 1
                if "sys_id=-1" in url or "sys_id%3d-1" in url:
                    c["qflag:sys_id_neg1"] += 3
            if el:
                for part in re.findall(r"/([a-z0-9]+)", el):
                    c[f"xtag:{part}"] += 1
        per_action.append(c)
    return per_action


def site_vocab(site_dicts_list: list[list[Counter]], min_sessions: int = 1) -> dict[str, int]:
    df: Counter = Counter()
    for per_action in site_dicts_list:
        seen = set()
        for c in per_action:
            seen.update(c.keys())
        df.update(seen)
    keep = sorted(k for k, v in df.items() if v >= min_sessions)
    return {k: i for i, k in enumerate(keep)}


def site_prefix_features(per_action: list[Counter], vocab: dict[str, int]) -> np.ndarray:
    """Cumulative schema-token indicators and counts at each action prefix k=1..n."""
    n = len(per_action)
    vlen = len(vocab)
    mat = np.zeros((n, vlen), dtype=np.float64)
    for a, c in enumerate(per_action):
        for k, val in c.items():
            col = vocab.get(k)
            if col is not None:
                mat[a, col] = val
    cum = np.cumsum(mat, axis=0)
    binary = (cum > 0).astype(np.float64)
    log_cum = np.log1p(cum)
    return np.hstack([binary, log_cum])


def assert_no_identity(names: list[str]) -> None:
    """Fail loudly if a feature name could carry page or run identity."""
    for name in names:
        low = name.lower()
        if any(tok in low for tok in ("url", "path", "host", "query", "agent", "task", "id:", "field:")):
            raise RuntimeError(f"identity-like feature name: {name}")
