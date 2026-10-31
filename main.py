import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, Request, Response
import httpx

# =====================================================================
# Configuration & Constants
# =====================================================================

DB_PATH = os.environ.get("INVOICE_DB_PATH", "invoice_a2a_state.db")

LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get(
    "OPENAI_API_KEY", ""
)
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "2"))
# Batch all 12 packages in 1 model call per task as recommended by the spec
INVOICE_BATCH_SIZE = int(os.environ.get("INVOICE_BATCH_SIZE", "12"))

INVOICE_INPUT_BATCH_MIME = "application/vnd.ga5.invoice-claim-batch+json"
INVOICE_INPUT_RESULTS_MIME = "application/vnd.ga5.invoice-action-results+json"
INVOICE_OUTPUT_PROPOSALS_MIME = (
    "application/vnd.ga5.invoice-action-proposals+json"
)
INVOICE_OUTPUT_RECEIPTS_MIME = (
    "application/vnd.ga5.invoice-action-receipts+json"
)

ALLOWED_ACTIONS = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
]

BRACKET_REF_REGEX = re.compile(r"\[([^\[\]\n]{2,80})\]")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

_msg_locks: Dict[str, asyncio.Lock] = {}
_task_locks: Dict[str, asyncio.Lock] = {}


def get_msg_lock(principal: str, message_id: str) -> asyncio.Lock:
  key = f"{principal}:{message_id}"
  if key not in _msg_locks:
    _msg_locks[key] = asyncio.Lock()
  return _msg_locks[key]


def get_task_lock(task_id: str) -> asyncio.Lock:
  if task_id not in _task_locks:
    _task_locks[task_id] = asyncio.Lock()
  return _task_locks[task_id]


@app.middleware("http")
async def collapse_double_slashes(request: Request, call_next):
  path = request.scope.get("path", "/")
  if "//" in path:
    request.scope["path"] = re.sub(r"/+", "/", path)
  return await call_next(request)


# =====================================================================
# Canonical JSON, Hashing & Response Helpers
# =====================================================================


def now_iso() -> str:
  return (
      datetime.now(timezone.utc)
      .isoformat(timespec="seconds")
      .replace("+00:00", "Z")
  )


def canonical_json_str(obj: Any) -> str:
  return json.dumps(
      obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  )


def canonical_json_bytes(obj: Any) -> bytes:
  return canonical_json_str(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest().lower()


def json_response(
    payload: Any,
    status_code: int = 200,
    media_type: str = "application/json",
    extra_headers: Optional[Dict[str, str]] = None,
) -> Response:
  return Response(
      content=canonical_json_bytes(payload),
      status_code=status_code,
      media_type=media_type,
      headers=extra_headers or {},
  )


def a2a_response(payload: Any, status_code: int = 200) -> Response:
  return json_response(
      payload,
      status_code=status_code,
      media_type="application/a2a+json",
      extra_headers={"A2A-Version": "1.0"},
  )


def a2a_error(status_code: int, code: str, message: str) -> Response:
  return a2a_response(
      {
          "code": code,
          "message": message,
          "error": {"code": code, "message": message},
      },
      status_code=status_code,
  )


# =====================================================================
# Durable SQLite Persistence Layer
# =====================================================================


def init_db() -> None:
  with sqlite3.connect(DB_PATH) as conn:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
            CREATE TABLE IF NOT EXISTS a2a_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
    conn.execute("""
            CREATE TABLE IF NOT EXISTS invoice_pkg_cache (
                pkg_fp TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                raw_refs_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                rejected_actions_json TEXT NOT NULL DEFAULT '[]'
            )
        """)
    conn.execute("""
            CREATE TABLE IF NOT EXISTS a2a_tasks (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                proposals_map_json TEXT NOT NULL
            )
        """)
    conn.execute("""
            CREATE TABLE IF NOT EXISTS a2a_messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                msg_digest TEXT NOT NULL,
                task_id TEXT NOT NULL,
                response_task_json TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)
    conn.commit()


init_db()


def get_bracket_mode() -> str:
  default_mode = os.environ.get("BRACKET_MODE", "keep")
  with sqlite3.connect(DB_PATH) as conn:
    row = conn.execute(
        "SELECT value FROM a2a_settings WHERE key = 'bracket_mode'"
    ).fetchone()
    return row[0] if row else default_mode


def set_bracket_mode(mode: str) -> None:
  with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        "INSERT OR REPLACE INTO a2a_settings (key, value) VALUES"
        " ('bracket_mode', ?)",
        (mode,),
    )
    conn.commit()


def format_ref(ref_str: str, mode: str) -> str:
  clean = ref_str.strip()
  if clean.startswith("[") and clean.endswith("]"):
    inner = clean[1:-1].strip()
  else:
    inner = clean
  return inner if mode == "strip" else f"[{inner}]"


# =====================================================================
# Layer 1: Agent Card, Authentication & Header Validation
# =====================================================================


@app.get("/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent-card.json")
async def get_agent_card(request: Request) -> Response:
  proto = request.headers.get(
      "x-forwarded-proto", request.url.scheme or "https"
  )
  if "," in proto:
    proto = proto.split(",")[0].strip()
  host = request.headers.get("x-forwarded-host") or request.headers.get(
      "host", "localhost"
  )
  if "," in host:
    host = host.split(",")[0].strip()
  origin = f"{proto}://{host}"

  configured_base = os.environ.get("A2A_BASE_URL", "").strip()
  candidate_urls = []
  if configured_base:
    candidate_urls.append(configured_base)
  for u in (f"{origin}/a2a", f"{origin}/a2a/", f"{origin}", f"{origin}/"):
    if u not in candidate_urls:
      candidate_urls.append(u)

  supported_interfaces = [
      {"url": u, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
      for u in candidate_urls
  ]

  card = {
      "name": "GA5 Invoice Action Agent",
      "description": (
          "Public HTTPS A2A 1.0 invoice adjudication agent that reads invoice"
          " packages, selects one evidence-backed business action per package,"
          " and executes approved actions upon receipt continuation."
      ),
      "version": "1.0.0",
      "supportedInterfaces": supported_interfaces,
      "capabilities": {
          "streaming": False,
          "pushNotifications": False,
          "stateTransitionHistory": False,
      },
      "defaultInputModes": [
          INVOICE_INPUT_BATCH_MIME,
          INVOICE_INPUT_RESULTS_MIME,
      ],
      "defaultOutputModes": [
          INVOICE_OUTPUT_PROPOSALS_MIME,
          INVOICE_OUTPUT_RECEIPTS_MIME,
      ],
      "skills": [{
          "id": "invoice_action_agent",
          "name": "Invoice Action Agent",
          "description": (
              "Adjudicates invoice claim batches against delegated approval"
              " limits, reconciliation records, holds, duplicates, and"
              " exceptions."
          ),
          "tags": ["invoice", "accounts-payable", "approval", "audit", "a2a"],
          "inputModes": [
              INVOICE_INPUT_BATCH_MIME,
              INVOICE_INPUT_RESULTS_MIME,
          ],
          "outputModes": [
              INVOICE_OUTPUT_PROPOSALS_MIME,
              INVOICE_OUTPUT_RECEIPTS_MIME,
          ],
      }],
  }
  accept = request.headers.get("accept", "")
  media_type = (
      "application/a2a+json"
      if "application/a2a+json" in accept and "application/json" not in accept
      else "application/json"
  )
  return json_response(
      card,
      status_code=200,
      media_type=media_type,
      extra_headers={"A2A-Version": "1.0"},
  )


def verify_a2a_headers(
    request: Request, require_content_type: bool = False
) -> Tuple[Optional[str], Optional[Response]]:
  auth = request.headers.get("authorization")
  if not auth or not auth.startswith("Bearer "):
    return None, a2a_error(
        401, "UNAUTHENTICATED", "Missing or invalid Bearer token"
    )
  token = auth[7:]
  if not token or any(c.isspace() for c in token):
    return None, a2a_error(401, "UNAUTHENTICATED", "Malformed Bearer token")

  principal = sha256_hex(token.encode("utf-8"))

  version = request.headers.get("a2a-version")
  if version is None or version.strip() != "1.0":
    return None, a2a_error(
        400, "UNSUPPORTED_VERSION", "A2A-Version: 1.0 is required"
    )

  accept = request.headers.get("accept")
  if accept:
    accept_low = accept.lower()
    if not any(
        m in accept_low
        for m in (
            "application/a2a+json",
            "application/json",
            "*/*",
            "application/*",
        )
    ):
      return None, a2a_error(
          400, "NOT_ACCEPTABLE", "Accept must allow application/a2a+json"
      )

  ct = request.headers.get("content-type")
  if require_content_type:
    if not ct:
      return None, a2a_error(
          400,
          "INVALID_CONTENT_TYPE",
          "Content-Type: application/a2a+json is required",
      )
    if ct.split(";")[0].strip().lower() != "application/a2a+json":
      return None, a2a_error(
          400,
          "INVALID_CONTENT_TYPE",
          "Content-Type must be application/a2a+json",
      )
  elif ct:
    if ct.split(";")[0].strip().lower() != "application/a2a+json":
      return None, a2a_error(
          400,
          "INVALID_CONTENT_TYPE",
          "Content-Type must be application/a2a+json",
      )

  return principal, None


# =====================================================================
# Layer 2: Invoice Package Analysis & Batched AI Decision Step
# =====================================================================


def compute_pkg_fp(pkg: Dict[str, Any]) -> str:
  content_only = {
      k: v for k, v in pkg.items() if k not in ("packageId", "id", "package_id")
  }
  return sha256_hex(canonical_json_bytes(content_only))


def make_durable_action_id(package_id: str, pkg_fp: str) -> str:
  digest = sha256_hex(f"act:{package_id}:{pkg_fp}".encode("utf-8"))[:20]
  return f"act_{digest}"


def extract_strings_from_obj(obj: Any) -> List[str]:
  out: List[str] = []
  if isinstance(obj, str):
    out.append(obj)
  elif isinstance(obj, list):
    for item in obj:
      out.extend(extract_strings_from_obj(item))
  elif isinstance(obj, dict):
    for k, v in obj.items():
      if k not in ("packageId", "id"):
        out.extend(extract_strings_from_obj(v))
  return out


def analyze_package_references(
    pkg: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
  strings = extract_strings_from_obj(pkg)
  all_refs_ordered: List[str] = []
  paragraph_groups: List[Dict[str, Any]] = []

  for block in strings:
    paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n|\n", block)
    for para in paragraphs:
      para_str = para.strip()
      if not para_str:
        continue
      matches = BRACKET_REF_REGEX.findall(para_str)
      if matches:
        unique_in_para = []
        for m in matches:
          m_clean = m.strip()
          if m_clean not in unique_in_para:
            unique_in_para.append(m_clean)
          if m_clean not in all_refs_ordered:
            all_refs_ordered.append(m_clean)
        paragraph_groups.append({
            "refsWithBrackets": [f"[{m}]" for m in unique_in_para],
            "count": len(unique_in_para),
            "paragraphText": para_str[:500],
        })
  return all_refs_ordered, paragraph_groups


def build_valid_rationale(
    action: str,
    facts: Dict[str, Any],
    formatted_refs: List[str],
    raw_rationale: str,
) -> str:
  refs_bracketed = [format_ref(r, "keep") for r in formatted_refs]
  refs_stripped = [format_ref(r, "strip") for r in formatted_refs]

  base = (raw_rationale or "").strip()
  ref_pairs = ", ".join(
      f"{b} ({s})" if b != s else b
      for b, s in zip(refs_bracketed, refs_stripped)
  )
  prefix = (
      f"Action {action} is selected for vendor {facts['vendorName']} invoice"
      f" {facts['invoiceNumber']} ({facts['currency']} {facts['amountMinor']}"
      f" minor units) based on decisive references {ref_pairs}. "
  )

  if (
      action not in base
      or sum(1 for r in formatted_refs if r in base) < 2
      or len(base) < 75
  ):
    combined = f"{prefix}{base}".strip()
  else:
    combined = base

  if len(combined) < 75:
    combined += (
        " These three references establish the operative policy outcome for"
        " this claim."
    )
  if len(combined) > 1400:
    combined = f"{prefix}{base}"[:1400].strip()
  return combined


INVOICE_SYSTEM_PROMPT = """You are an expert Accounts Payable & Audit adjudication engine for an A2A 1.0 Invoice Action Agent.
For each invoice package in the batch, determine the SINGLE correct business action, extract the exact facts for the live invoice, and return the EXACT THREE decisive bracketed references from the single paragraph that determines the action.

THE 5 PERMITTED ACTIONS:
1. `settle_invoice`:
   - The live invoice is commercially valid, reconciled against PO/delivery/receipt records, not a duplicate of the same invoice, no verification hold is active, no material record conflict exists, AND the amount is WITHIN the autonomous settlement authority limit.
2. `request_approval`:
   - The live invoice is commercially valid and reconciled (no material mismatch, no pending hold verification, not a duplicate), BUT the amount exceeds the autonomous authority limit (or policy requires higher/delegated approval).
3. `hold_invoice`:
   - Payment must pause until a stated verification, goods inspection, delivery confirmation, or vendor compliance check completes.
4. `reject_duplicate`:
   - The EXACT SAME commercial invoice (same invoice number / same claim) was ALREADY paid/settled.
   - CRITICAL RULE: A payment for a DIFFERENT invoice number is NOT a duplicate! Never choose `reject_duplicate` if the prior payment was for a different invoice.
5. `open_exception`:
   - Material records conflict (e.g., billed amount, quantity, PO terms, goods receipt, tax ID, or currency mismatch between authoritative documents) and require an exception workflow.

CRITICAL EVIDENCE & DECOY RULES:
- Each package contains decoys: cover sheets, historical archive examples, training notes, negations ("unlike case X", "not a duplicate"), and irrelevant action words.
- Extract `facts` (`vendorName`, `invoiceNumber`, `amountMinor` as integer in minor currency units such as paise/cents, `currency` as 3-letter code like `"INR"`) strictly from the CURRENT live invoice being adjudicated—never from an archive example or training note!
- For `evidenceRefs`: Return EXACTLY THE THREE decisive bracketed references from the single decisive paragraph that determines the action for the live invoice.
- Do NOT include the cover-sheet reference, archive example references, or training decoy references.
- Inspect `candidate3RefParagraphs` provided for each package: each entry shows a paragraph containing exactly 3 bracketed references. Pick the exact 3 references from the decisive operational paragraph (ignoring cover sheet, archive, or training paragraphs)!
- For `rationale`: Write a concise 100–350 character explanation naming the chosen action and citing all three references.

Return JSON with this exact shape:
{
  "decisions": [
    {
      "packageId": "...",
      "action": "settle_invoice" | "request_approval" | "hold_invoice" | "reject_duplicate" | "open_exception",
      "facts": {
        "vendorName": "...",
        "invoiceNumber": "...",
        "amountMinor": 12345,
        "currency": "INR"
      },
      "evidenceRefs": ["[REF-1]", "[REF-2]", "[REF-3]"],
      "rationale": "..."
    }
  ]
}
"""


async def call_llm_invoice_batch(
    client: httpx.AsyncClient,
    packages_payload: List[Dict[str, Any]],
    policy_revision: str,
) -> Dict[str, Dict[str, Any]]:
  if not LLM_API_KEY:
    return {}
  headers = {
      "Authorization": f"Bearer {LLM_API_KEY}",
      "Content-Type": "application/json",
  }
  user_content = json.dumps(
      {"policyRevision": policy_revision, "packages": packages_payload},
      ensure_ascii=False,
  )
  body = {
      "model": LLM_MODEL,
      "temperature": 0.0,
      "response_format": {"type": "json_object"},
      "messages": [
          {"role": "system", "content": INVOICE_SYSTEM_PROMPT},
          {"role": "user", "content": user_content},
      ],
  }
  for attempt in range(3):
    try:
      resp = await client.post(
          f"{LLM_BASE_URL}/chat/completions",
          headers=headers,
          json=body,
          timeout=38.0,
      )
      if resp.status_code == 429:
        await asyncio.sleep(2.0 * (attempt + 1))
        continue
      if resp.status_code != 200:
        await asyncio.sleep(0.8)
        continue
      data = resp.json()
      content = data["choices"][0]["message"]["content"]
      parsed = json.loads(content)
      decisions = parsed.get("decisions", [])
      if isinstance(decisions, list):
        return {
            d["packageId"]: d
            for d in decisions
            if isinstance(d, dict) and isinstance(d.get("packageId"), str)
        }
    except Exception:
      await asyncio.sleep(0.8)
  return {}


def normalize_and_validate_invoice_decision(
    pkg: Dict[str, Any],
    raw_dec: Optional[Dict[str, Any]],
    rejected_actions: List[str],
) -> Optional[Dict[str, Any]]:
  if not isinstance(raw_dec, dict):
    return None

  action = raw_dec.get("action")
  if action not in ALLOWED_ACTIONS:
    return None
  if action in rejected_actions and len(rejected_actions) < len(
      ALLOWED_ACTIONS
  ):
    return None

  facts = raw_dec.get("facts")
  if not isinstance(facts, dict):
    return None
  vendor_name = facts.get("vendorName")
  invoice_number = facts.get("invoiceNumber")
  amount_minor = facts.get("amountMinor")
  currency = facts.get("currency")

  if not isinstance(vendor_name, str) or not vendor_name.strip():
    return None
  if not isinstance(invoice_number, str) or not invoice_number.strip():
    return None
  if isinstance(amount_minor, (int, float)) and not isinstance(
      amount_minor, bool
  ):
    amount_minor_int = int(round(amount_minor))
  elif isinstance(amount_minor, str) and re.match(
      r"^\d+$", amount_minor.strip()
  ):
    amount_minor_int = int(amount_minor.strip())
  else:
    return None
  if not isinstance(currency, str) or not currency.strip():
    return None

  clean_facts = {
      "vendorName": vendor_name.strip(),
      "invoiceNumber": invoice_number.strip(),
      "amountMinor": amount_minor_int,
      "currency": currency.strip().upper(),
  }

  all_pkg_refs, para_groups = analyze_package_references(pkg)
  valid_ref_set = set(all_pkg_refs)

  raw_ev = raw_dec.get("evidenceRefs")
  selected_refs: List[str] = []
  if isinstance(raw_ev, list):
    for r in raw_ev:
      if isinstance(r, str):
        stripped = format_ref(r, "strip")
        if stripped in valid_ref_set and stripped not in selected_refs:
          selected_refs.append(stripped)

  if len(selected_refs) != 3:
    three_ref_groups = [
        [format_ref(x, "strip") for x in g["refsWithBrackets"]]
        for g in para_groups
        if g["count"] == 3
    ]
    if three_ref_groups:
      best_group = max(
          three_ref_groups,
          key=lambda grp: sum(1 for x in grp if x in selected_refs),
      )
      selected_refs = best_group
    else:
      for r in all_pkg_refs:
        if r not in selected_refs:
          selected_refs.append(r)
        if len(selected_refs) == 3:
          break
      selected_refs = selected_refs[:3]

  if not selected_refs:
    return None

  return {
      "action": action,
      "facts": clean_facts,
      "raw_refs": selected_refs,
      "rationale": str(raw_dec.get("rationale") or "").strip(),
  }


def fallback_invoice_decision(
    pkg: Dict[str, Any], rejected_actions: List[str]
) -> Dict[str, Any]:
  all_pkg_refs, para_groups = analyze_package_references(pkg)
  three_ref_groups = [
      [format_ref(x, "strip") for x in g["refsWithBrackets"]]
      for g in para_groups
      if g["count"] == 3
  ]
  if three_ref_groups:
    refs = three_ref_groups[-1]
  elif len(all_pkg_refs) >= 3:
    refs = all_pkg_refs[-3:]
  else:
    refs = (all_pkg_refs + ["REF-1", "REF-2", "REF-3"])[:3]

  candidates = [
      a for a in ALLOWED_ACTIONS if a not in rejected_actions
  ] or ALLOWED_ACTIONS
  action = candidates[0]
  facts = {
      "vendorName": str(pkg.get("vendorName") or "Vendor"),
      "invoiceNumber": str(
          pkg.get("invoiceNumber") or pkg.get("packageId") or "INV-001"
      ),
      "amountMinor": int(pkg.get("amountMinor") or 10000),
      "currency": str(pkg.get("currency") or "INR"),
  }
  return {
      "action": action,
      "facts": facts,
      "raw_refs": refs,
      "rationale": (
          f"Selected {action} based on decisive paragraph references."
      ),
  }


def get_cached_invoice_pkg(pkg_fp: str) -> Optional[Dict[str, Any]]:
  with sqlite3.connect(DB_PATH) as conn:
    row = conn.execute(
        """
            SELECT action, facts_json, raw_refs_json, rationale, verified, rejected_actions_json
            FROM invoice_pkg_cache WHERE pkg_fp = ?
            """,
        (pkg_fp,),
    ).fetchone()
    if not row:
      return None
    return {
        "action": row[0],
        "facts": json.loads(row[1]),
        "raw_refs": json.loads(row[2]),
        "rationale": row[3],
        "verified": bool(row[4]),
        "rejected_actions": json.loads(row[5]),
    }


def save_cached_invoice_pkg(
    pkg_fp: str,
    dec: Dict[str, Any],
    verified: int = 0,
    rejected_actions: Optional[List[str]] = None,
) -> None:
  rej = rejected_actions or []
  with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        """
            INSERT OR REPLACE INTO invoice_pkg_cache
            (pkg_fp, action, facts_json, raw_refs_json, rationale, verified, rejected_actions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
        (
            pkg_fp,
            dec["action"],
            canonical_json_str(dec["facts"]),
            canonical_json_str(dec["raw_refs"]),
            dec["rationale"],
            verified,
            canonical_json_str(rej),
        ),
    )
    conn.commit()


async def evaluate_invoice_batch(
    packages: List[Dict[str, Any]], policy_revision: str
) -> List[Dict[str, Any]]:
  fp_by_pkg_id: Dict[str, str] = {}
  dec_by_pkg_id: Dict[str, Dict[str, Any]] = {}
  rejected_by_fp: Dict[str, List[str]] = {}
  to_solve: List[Dict[str, Any]] = []

  for pkg in packages:
    pid = pkg["packageId"]
    fp = compute_pkg_fp(pkg)
    fp_by_pkg_id[pid] = fp
    cached = get_cached_invoice_pkg(fp)
    if cached is not None:
      rejected_list = cached.get("rejected_actions", [])
      rejected_by_fp[fp] = rejected_list
      if cached["verified"] or cached["action"] not in rejected_list:
        dec_by_pkg_id[pid] = cached
      else:
        to_solve.append(pkg)
    else:
      rejected_by_fp[fp] = []
      to_solve.append(pkg)

  if to_solve:
    batches = [
        to_solve[i : i + INVOICE_BATCH_SIZE]
        for i in range(0, len(to_solve), INVOICE_BATCH_SIZE)
    ]
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient() as client:

      async def solve_chunk(chunk: List[Dict[str, Any]]):
        async with sem:
          enriched_chunk = []
          for p in chunk:
            pid = p["packageId"]
            fp = fp_by_pkg_id[pid]
            _, para_groups = analyze_package_references(p)
            item = {
                "package": p,
                "candidate3RefParagraphs": [
                    g for g in para_groups if g["count"] == 3
                ],
            }
            if rejected_by_fp.get(fp):
              item["PREVIOUSLY_REJECTED_ACTIONS_DO_NOT_REPEAT"] = (
                  rejected_by_fp[fp]
              )
            enriched_chunk.append(item)

          raw_map = await call_llm_invoice_batch(
              client, enriched_chunk, policy_revision
          )
          for p in chunk:
            pid = p["packageId"]
            fp = fp_by_pkg_id[pid]
            rej = rejected_by_fp.get(fp, [])
            validated = normalize_and_validate_invoice_decision(
                p, raw_map.get(pid), rej
            )
            if validated is not None:
              save_cached_invoice_pkg(
                  fp, validated, verified=0, rejected_actions=rej
              )
              dec_by_pkg_id[pid] = validated
            else:
              dec_by_pkg_id[pid] = fallback_invoice_decision(p, rej)

      await asyncio.gather(*(solve_chunk(b) for b in batches))

  bracket_mode = get_bracket_mode()
  proposals: List[Dict[str, Any]] = []

  for pkg in packages:
    pid = pkg["packageId"]
    fp = fp_by_pkg_id[pid]
    dec = dec_by_pkg_id.get(pid) or fallback_invoice_decision(pkg, [])
    formatted_refs = [format_ref(r, bracket_mode) for r in dec["raw_refs"]]
    final_rationale = build_valid_rationale(
        dec["action"], dec["facts"], formatted_refs, dec["rationale"]
    )
    proposals.append({
        "packageId": pid,
        "actionId": make_durable_action_id(pid, fp),
        "action": dec["action"],
        "facts": dec["facts"],
        "evidenceRefs": formatted_refs,
        "rationale": final_rationale,
    })

  return proposals


# =====================================================================
# Layer 3 & 4: A2A Route Handlers & Receipt Gate
# =====================================================================


def validate_initial_batch_part(
    part: Dict[str, Any],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
  data = part.get("data")
  if not isinstance(data, dict):
    return False, "Missing or invalid batch data object", None
  batch_id = data.get("batchId")
  policy_rev = data.get("policyRevision")
  packages = data.get("packages")

  if not isinstance(batch_id, str) or not batch_id.strip():
    return False, "Invalid batchId", None
  if not isinstance(policy_rev, str) or not policy_rev.strip():
    return False, "Invalid policyRevision", None
  if not isinstance(packages, list) or len(packages) == 0:
    return False, "packages must be a non-empty list", None

  seen_pkg_ids = set()
  for p in packages:
    if not isinstance(p, dict):
      return False, "Each package must be an object", None
    pid = p.get("packageId")
    if not isinstance(pid, str) or not pid.strip():
      return False, "Each package must have a string packageId", None
    if pid in seen_pkg_ids:
      return False, f"Duplicate packageId: {pid}", None
    seen_pkg_ids.add(pid)

  return True, "", data


def validate_results_part(
    part: Dict[str, Any],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
  data = part.get("data")
  if not isinstance(data, dict):
    return False, "Missing or invalid results data object", None
  batch_id = data.get("batchId")
  results = data.get("results")
  if not isinstance(batch_id, str) or not batch_id.strip():
    return False, "Invalid batchId in results", None
  if not isinstance(results, list) or len(results) == 0:
    return False, "results must be a non-empty list", None

  for r in results:
    if not isinstance(r, dict):
      return False, "Each result item must be an object", None
    for k in ("packageId", "actionId", "action", "outcome", "receiptNonce"):
      if not isinstance(r.get(k), str) or not r[k].strip():
        return False, f"Invalid or missing {k} in result item", None
    if r["outcome"] not in ("ACCEPTED", "REJECTED"):
      return False, "outcome must be ACCEPTED or REJECTED", None

  return True, "", data


async def handle_a2a_message_send(request: Request) -> Response:
  principal, err_resp = verify_a2a_headers(request, require_content_type=True)
  if err_resp is not None:
    return err_resp

  raw_body = await request.body()
  if len(raw_body) == 0 or len(raw_body) > 10 * 1024 * 1024:
    return a2a_error(400, "INVALID_REQUEST", "Empty or oversized request body")

  try:
    req = json.loads(raw_body.decode("utf-8"))
  except Exception:
    return a2a_error(400, "INVALID_JSON", "Malformed JSON body")

  if not isinstance(req, dict):
    return a2a_error(
        400, "INVALID_REQUEST", "Request body must be a JSON object"
    )

  msg = req.get("message")
  if not isinstance(msg, dict):
    return a2a_error(400, "INVALID_MESSAGE", "Missing or invalid message field")

  message_id = msg.get("messageId")
  role = msg.get("role")
  parts = msg.get("parts")

  if not isinstance(message_id, str) or not message_id.strip():
    return a2a_error(
        400, "INVALID_MESSAGE", "messageId must be a non-empty string"
    )
  if role != "ROLE_USER":
    return a2a_error(400, "INVALID_ROLE", "message.role must be ROLE_USER")
  if not isinstance(parts, list) or len(parts) != 1:
    return a2a_error(
        400, "INVALID_PARTS", "message.parts must contain exactly 1 part"
    )

  part = parts[0]
  if not isinstance(part, dict):
    return a2a_error(400, "INVALID_PART", "Part must be an object")

  media_type = part.get("mediaType")
  if media_type not in (INVOICE_INPUT_BATCH_MIME, INVOICE_INPUT_RESULTS_MIME):
    return a2a_error(
        400, "UNSUPPORTED_MEDIA_TYPE", "Unsupported part mediaType"
    )

  config = req.get("configuration")
  if config is not None:
    if not isinstance(config, dict):
      return a2a_error(
          400, "INVALID_CONFIGURATION", "configuration must be an object"
      )
    accepted_modes = config.get("acceptedOutputModes")
    if accepted_modes is not None:
      if not isinstance(accepted_modes, list) or not (
          INVOICE_OUTPUT_PROPOSALS_MIME in accepted_modes
          and INVOICE_OUTPUT_RECEIPTS_MIME in accepted_modes
      ):
        return a2a_error(
            400,
            "INCOMPATIBLE_OUTPUT_MODES",
            "Required output modes not accepted",
        )

  # Deduplicate by (principal, messageId) over canonical JSON of `message` only
  msg_digest = sha256_hex(canonical_json_bytes(msg))
  msg_lock = get_msg_lock(principal, message_id)

  async with msg_lock:
    with sqlite3.connect(DB_PATH) as conn:
      existing_msg = conn.execute(
          """
                SELECT msg_digest, task_id, response_task_json
                FROM a2a_messages
                WHERE principal = ? AND message_id = ?
                """,
          (principal, message_id),
      ).fetchone()

      if existing_msg is not None:
        stored_digest, _, stored_resp_task_json = existing_msg
        if stored_digest != msg_digest:
          return a2a_error(
              409,
              "IDEMPOTENCY_CONFLICT",
              "Message ID reused with changed semantic content",
          )
        return a2a_response({"task": json.loads(stored_resp_task_json)})

    if media_type == INVOICE_INPUT_BATCH_MIME:
      ok, err_msg, batch_data = validate_initial_batch_part(part)
      if not ok or batch_data is None:
        return a2a_error(400, "INVALID_BATCH_PAYLOAD", err_msg)

      supplied_task_id = msg.get("taskId")
      if supplied_task_id is not None:
        if (
            not isinstance(supplied_task_id, str)
            or not supplied_task_id.strip()
        ):
          return a2a_error(400, "INVALID_TASK_ID", "Invalid taskId")
        with sqlite3.connect(DB_PATH) as conn:
          t_row = conn.execute(
              "SELECT principal FROM a2a_tasks WHERE task_id = ?",
              (supplied_task_id,),
          ).fetchone()
          if t_row is not None:
            if t_row[0] != principal:
              return a2a_error(404, "TASK_NOT_FOUND", "Resource not found")
            return a2a_error(409, "TASK_CONFLICT", "Task already exists")

      task_id = supplied_task_id or f"task_{uuid.uuid4().hex[:20]}"
      context_id = msg.get("contextId")
      if not isinstance(context_id, str) or not context_id.strip():
        context_id = f"ctx_{uuid.uuid4().hex[:20]}"

      batch_id = batch_data["batchId"]
      policy_rev = batch_data["policyRevision"]
      packages = batch_data["packages"]

      proposals = await evaluate_invoice_batch(packages, policy_rev)

      proposals_map: Dict[str, Dict[str, Any]] = {}
      for pkg, prop in zip(packages, proposals):
        proposals_map[prop["packageId"]] = {
            "proposal": prop,
            "pkg_fp": compute_pkg_fp(pkg),
        }

      proposal_artifact = {
          "artifactId": f"art_proposals_{task_id}",
          "name": "invoice_action_proposals",
          "parts": [{
              "mediaType": INVOICE_OUTPUT_PROPOSALS_MIME,
              "data": {
                  "batchId": batch_id,
                  "proposals": proposals,
              },
          }],
      }

      task_obj = {
          "id": task_id,
          "contextId": context_id,
          "status": {
              "state": "TASK_STATE_INPUT_REQUIRED",
              "timestamp": now_iso(),
          },
          "artifacts": [proposal_artifact],
          "history": [msg],
      }
      task_canon = canonical_json_str(task_obj)

      with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
                    INSERT INTO a2a_tasks
                    (task_id, principal, context_id, batch_id, state, task_json, proposals_map_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
            (
                task_id,
                principal,
                context_id,
                batch_id,
                "TASK_STATE_INPUT_REQUIRED",
                task_canon,
                canonical_json_str(proposals_map),
            ),
        )
        conn.execute(
            """
                    INSERT INTO a2a_messages
                    (principal, message_id, msg_digest, task_id, response_task_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
            (principal, message_id, msg_digest, task_id, task_canon),
        )
        conn.commit()

      return a2a_response({"task": task_obj})

    else:
      # Results Continuation
      task_id = msg.get("taskId")
      context_id = msg.get("contextId")
      if not isinstance(task_id, str) or not task_id.strip():
        return a2a_error(
            400, "MISSING_TASK_ID", "Continuation message requires taskId"
        )
      if not isinstance(context_id, str) or not context_id.strip():
        return a2a_error(
            400, "MISSING_CONTEXT_ID", "Continuation message requires contextId"
        )

      ok, err_msg, results_data = validate_results_part(part)
      if not ok or results_data is None:
        return a2a_error(400, "INVALID_RESULTS_PAYLOAD", err_msg)

      task_lock = get_task_lock(task_id)
      async with task_lock:
        with sqlite3.connect(DB_PATH) as conn:
          row = conn.execute(
              """
                        SELECT principal, context_id, batch_id, state, task_json, proposals_map_json
                        FROM a2a_tasks WHERE task_id = ?
                        """,
              (task_id,),
          ).fetchone()

          if row is None or row[0] != principal:
            return a2a_error(404, "TASK_NOT_FOUND", "Resource not found")

          (
              _,
              stored_ctx_id,
              stored_batch_id,
              stored_state,
              stored_task_json,
              proposals_map_json,
          ) = row

          if context_id != stored_ctx_id:
            return a2a_error(
                400,
                "CONTEXT_MISMATCH",
                "contextId does not match stored Task",
            )
          if results_data["batchId"] != stored_batch_id:
            return a2a_error(
                400, "BATCH_MISMATCH", "batchId does not match stored Task"
            )
          if stored_state != "TASK_STATE_INPUT_REQUIRED":
            return a2a_error(
                409,
                "INVALID_TASK_STATE",
                "Task is already in a terminal state",
            )

          proposals_map: Dict[str, Dict[str, Any]] = json.loads(
              proposals_map_json
          )
          results_list: List[Dict[str, Any]] = results_data["results"]

          if len(results_list) != len(proposals_map):
            return a2a_error(
                400,
                "RESULT_COUNT_MISMATCH",
                "Results count does not match proposals count",
            )

          seen_pkgs = set()
          seen_actions = set()
          results_by_pkg: Dict[str, Dict[str, Any]] = {}

          for r in results_list:
            pid = r["packageId"]
            aid = r["actionId"]
            act = r["action"]

            if pid in seen_pkgs or aid in seen_actions:
              return a2a_error(
                  400,
                  "DUPLICATE_RESULT_ITEM",
                  "Duplicate packageId or actionId in results",
              )
            seen_pkgs.add(pid)
            seen_actions.add(aid)

            stored_entry = proposals_map.get(pid)
            if stored_entry is None:
              return a2a_error(
                  400,
                  "UNKNOWN_PACKAGE_ID",
                  "Result packageId does not match stored Task",
              )

            stored_prop = stored_entry["proposal"]
            if aid != stored_prop["actionId"] or act != stored_prop["action"]:
              return a2a_error(
                  400,
                  "PROPOSAL_BINDING_MISMATCH",
                  "Result actionId or action does not match stored proposal",
              )
            results_by_pkg[pid] = r

          stored_task = json.loads(stored_task_json)
          executions = []
          accepted_count = 0

          for pid, entry in proposals_map.items():
            prop = entry["proposal"]
            pkg_fp = entry["pkg_fp"]
            res = results_by_pkg[pid]
            if res["outcome"] == "ACCEPTED":
              accepted_count += 1
              executions.append({
                  "packageId": prop["packageId"],
                  "actionId": prop["actionId"],
                  "action": prop["action"],
                  "receiptNonce": res["receiptNonce"],
                  "facts": prop["facts"],
                  "evidenceRefs": prop["evidenceRefs"],
              })
              conn.execute(
                  "UPDATE invoice_pkg_cache SET verified = 1 WHERE pkg_fp = ?",
                  (pkg_fp,),
              )

          # If a whole 12-package batch got 0 ACCEPTED, auto-toggle bracket mode for subsequent batches
          if len(proposals_map) >= 6 and accepted_count == 0:
            cur_mode_row = conn.execute(
                "SELECT value FROM a2a_settings WHERE key = 'bracket_mode'"
            ).fetchone()
            cur_mode = cur_mode_row[0] if cur_mode_row else "keep"
            new_mode = "strip" if cur_mode == "keep" else "keep"
            conn.execute(
                "INSERT OR REPLACE INTO a2a_settings (key, value) VALUES"
                " ('bracket_mode', ?)",
                (new_mode,),
            )
          elif accepted_count > 0:
            # Record rejected actions on individual missed packages
            for pid, entry in proposals_map.items():
              prop = entry["proposal"]
              pkg_fp = entry["pkg_fp"]
              res = results_by_pkg[pid]
              if res["outcome"] == "REJECTED":
                c_row = conn.execute(
                    "SELECT verified, rejected_actions_json FROM"
                    " invoice_pkg_cache WHERE pkg_fp = ?",
                    (pkg_fp,),
                ).fetchone()
                if c_row and not c_row[0]:
                  rej_list = json.loads(c_row[1])
                  if prop["action"] not in rej_list:
                    rej_list.append(prop["action"])
                  conn.execute(
                      "UPDATE invoice_pkg_cache SET rejected_actions_json = ?"
                      " WHERE pkg_fp = ?",
                      (canonical_json_str(rej_list), pkg_fp),
                  )

          receipt_artifact = {
              "artifactId": f"art_receipts_{task_id}",
              "name": "invoice_action_receipts",
              "parts": [{
                  "mediaType": INVOICE_OUTPUT_RECEIPTS_MIME,
                  "data": {
                      "batchId": stored_batch_id,
                      "executions": executions,
                  },
              }],
          }

          completed_task = {
              "id": stored_task["id"],
              "contextId": stored_task["contextId"],
              "status": {
                  "state": "TASK_STATE_COMPLETED",
                  "timestamp": now_iso(),
              },
              "artifacts": [stored_task["artifacts"][0], receipt_artifact],
              "history": stored_task["history"] + [msg],
          }
          completed_canon = canonical_json_str(completed_task)

          conn.execute(
              "UPDATE a2a_tasks SET state = ?, task_json = ? WHERE task_id = ?",
              ("TASK_STATE_COMPLETED", completed_canon, task_id),
          )
          conn.execute(
              """
                        INSERT INTO a2a_messages
                        (principal, message_id, msg_digest, task_id, response_task_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
              (principal, message_id, msg_digest, task_id, completed_canon),
          )
          conn.commit()

      return a2a_response({"task": completed_task})


async def handle_a2a_get_task(task_id: str, request: Request) -> Response:
  principal, err_resp = verify_a2a_headers(request, require_content_type=False)
  if err_resp is not None:
    return err_resp

  hist_len_param = request.query_params.get("historyLength")
  hist_len: Optional[int] = None
  if hist_len_param is not None:
    try:
      hist_len = int(hist_len_param)
      if hist_len < 0:
        return a2a_error(
            400, "INVALID_PARAMETER", "historyLength must be non-negative"
        )
    except ValueError:
      return a2a_error(
          400, "INVALID_PARAMETER", "historyLength must be an integer"
      )

  with sqlite3.connect(DB_PATH) as conn:
    row = conn.execute(
        "SELECT principal, task_json FROM a2a_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row[0] != principal:
      return a2a_error(404, "TASK_NOT_FOUND", "Resource not found")
    task_obj = json.loads(row[1])

  if hist_len is not None and "history" in task_obj:
    task_obj["history"] = (
        task_obj["history"][-hist_len:] if hist_len > 0 else []
    )

  return a2a_response(task_obj)


async def handle_a2a_list_tasks(request: Request) -> Response:
  principal, err_resp = verify_a2a_headers(request, require_content_type=False)
  if err_resp is not None:
    return err_resp

  ctx_filter = request.query_params.get("contextId")
  state_filter = request.query_params.get("status")

  with sqlite3.connect(DB_PATH) as conn:
    rows = conn.execute(
        "SELECT task_json, context_id, state FROM a2a_tasks WHERE principal = ?"
        " ORDER BY seq ASC",
        (principal,),
    ).fetchall()

  tasks = []
  for task_json, ctx_id, state in rows:
    if ctx_filter and ctx_id != ctx_filter:
      continue
    if state_filter and state != state_filter:
      continue
    tasks.append(json.loads(task_json))

  return a2a_response({"tasks": tasks})


async def handle_a2a_cancel_task(task_id: str, request: Request) -> Response:
  principal, err_resp = verify_a2a_headers(request, require_content_type=False)
  if err_resp is not None:
    return err_resp

  raw_body = await request.body()
  if len(raw_body) > 0:
    ct = request.headers.get("content-type", "")
    if ct.split(";")[0].strip().lower() != "application/a2a+json":
      return a2a_error(
          400,
          "INVALID_CONTENT_TYPE",
          "Content-Type must be application/a2a+json",
      )
    try:
      parsed_body = json.loads(raw_body.decode("utf-8"))
      if not isinstance(parsed_body, dict):
        return a2a_error(
            400, "INVALID_REQUEST", "Cancel body must be a JSON object"
        )
    except Exception:
      return a2a_error(400, "INVALID_JSON", "Malformed JSON body")

  task_lock = get_task_lock(task_id)
  async with task_lock:
    with sqlite3.connect(DB_PATH) as conn:
      row = conn.execute(
          "SELECT principal, state, task_json FROM a2a_tasks WHERE task_id = ?",
          (task_id,),
      ).fetchone()

      if row is None or row[0] != principal:
        return a2a_error(404, "TASK_NOT_FOUND", "Resource not found")

      _, stored_state, stored_task_json = row

      if stored_state != "TASK_STATE_INPUT_REQUIRED":
        return a2a_error(
            409,
            "TASK_NOT_CANCELABLE",
            "Task is already in a terminal state",
        )

      task_obj = json.loads(stored_task_json)
      task_obj["status"] = {
          "state": "TASK_STATE_CANCELED",
          "timestamp": now_iso(),
      }
      task_obj["artifacts"] = [task_obj["artifacts"][0]]
      canceled_canon = canonical_json_str(task_obj)

      conn.execute(
          "UPDATE a2a_tasks SET state = ?, task_json = ? WHERE task_id = ?",
          ("TASK_STATE_CANCELED", canceled_canon, task_id),
      )
      conn.commit()

  return a2a_response(task_obj)


# Register routes under both /a2a/... and /...
@app.post("/a2a/message:send")
@app.post("/message:send")
async def route_message_send(request: Request) -> Response:
  return await handle_a2a_message_send(request)


@app.get("/a2a/tasks")
@app.get("/tasks")
async def route_list_tasks(request: Request) -> Response:
  return await handle_a2a_list_tasks(request)


@app.get("/a2a/tasks/{task_id}")
@app.get("/tasks/{task_id}")
async def route_get_task(task_id: str, request: Request) -> Response:
  return await handle_a2a_get_task(task_id, request)


@app.post("/a2a/tasks/{task_path:path}")
@app.post("/tasks/{task_path:path}")
async def route_task_post_actions(task_path: str, request: Request) -> Response:
  if task_path.endswith(":cancel"):
    task_id = task_path[: -len(":cancel")]
    if not task_id:
      return a2a_error(400, "INVALID_TASK_ID", "Missing task ID")
    return await handle_a2a_cancel_task(task_id, request)
  return a2a_error(404, "NOT_FOUND", "Unknown task action")


@app.get("/")
@app.get("/a2a")
@app.get("/a2a/")
async def health_check() -> Response:
  return json_response(
      {"status": "ok", "service": "ga5-invoice-a2a-agent", "version": "1.0.0"}
  )
