import hashlib
import json
import uuid
import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, Header, Depends, HTTPException

# --- Examiner Visibility Setup ---
# This ensures the examiner sees your AI's thought process in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI()

# --- In-Memory State ---
db_tasks: Dict[str, Dict[str, Any]] = {}
db_task_owners: Dict[str, str] = {}
db_idempotency: Dict[str, Dict[str, str]] = {}
task_locks: Dict[str, asyncio.Lock] = {}

def save_state_for_examiner():
    """Saves the current tasks to a JSON file so the examiner can grade it offline."""
    with open("offline_grader_state.json", "w") as f:
        json.dump(db_tasks, f, indent=2)
    logging.info("Offline state saved to offline_grader_state.json")

# --- Protocol Dependencies ---
def get_auth_token(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    return authorization.split(" ")[1]

def enforce_a2a_headers(
    a2a_version: str = Header(None, alias="A2A-Version"),
    content_type: str = Header(None, alias="Content-Type")
):
    if a2a_version != "1.0":
        raise HTTPException(status_code=400, detail="A2A-Version must be 1.0")
    if content_type and content_type != "application/a2a+json":
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

def hash_message(message_dict: dict) -> str:
    compact_json = json.dumps(message_dict, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(compact_json.encode('utf-8')).hexdigest()

# --- Public Agent Card ---
@app.get("/.well-known/agent-card.json")
async def get_agent_card(request: Request):
    base_url = str(request.base_url).rstrip('/')
    return {
        "name": "Invoice Action Agent",
        "description": "Evaluates invoice packages and securely proposes business actions.",
        "version": "1.0.0",
        "capabilities": {},
        "skills": {
            "invoice_action_agent": {
                "name": "Invoice AI Analyst",
                "description": "Analyzes invoices for settlement, holding, or rejection.",
                "tags": ["invoice", "finance"]
            }
        },
        "supportedInterfaces": [{
            "url": f"{base_url}/a2a",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }

# --- Core Task Processing ---
@app.post("/a2a/message:send")
async def send_message(request: Request, token: str = Depends(get_auth_token), _=Depends(enforce_a2a_headers)):
    payload = await request.json()
    message = payload.get("message", {})
    message_id = message.get("messageId")
    
    if not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId")
    
    msg_hash = hash_message(message)
    idem_key = f"{token}:{message_id}"
    
    # 1. Idempotency Check
    if idem_key in db_idempotency:
        stored = db_idempotency[idem_key]
        if stored["hash"] != msg_hash:
            raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
        return {"task": db_tasks[stored["task_id"]]}
    
    parts = message.get("parts", [])
    if not parts:
        raise HTTPException(status_code=400, detail="Missing parts")
    
    media_type = parts[0].get("mediaType")
    data = parts[0].get("data", {})
    
    # PHASE 1: Proposal Generation
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        task_id = f"task_{uuid.uuid4().hex}"
        db_task_owners[task_id] = token
        
        logging.info(f"\n--- NEW INVOICE BATCH RECEIVED (Task: {task_id}) ---")
        
        proposals = []
        for pkg in data.get("packages", []):
            pkg_id = pkg.get("packageId", "unknown")
            
            # --- AI LOGIC GOES HERE ---
            # Replace these static variables with a call to your AI model
            action = "request_approval" 
            rationale = "Amount exceeds limit. Requires manual review."
            evidence = ["Exact quote 1", "Exact quote 2", "Exact quote 3"]
            
            logging.info(f"[Package: {pkg_id}] AI chose Action: '{action}'")
            logging.info(f"[Package: {pkg_id}] AI Rationale: {rationale}")
            
            proposals.append({
                "packageId": pkg_id,
                "actionId": f"act_{uuid.uuid4().hex}",
                "action": action,
                "facts": {"vendorName": "AI Vendor", "invoiceNumber": "INV-01", "amountMinor": 1000, "currency": "USD"},
                "evidenceRefs": evidence,
                "rationale": rationale
            })
            
        task_obj = {
            "taskId": task_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {"batchId": data.get("batchId"), "proposals": proposals}
            }]
        }
        
        db_tasks[task_id] = task_obj
        db_idempotency[idem_key] = {"hash": msg_hash, "task_id": task_id}
        task_locks[task_id] = asyncio.Lock()
        
        save_state_for_examiner()
        return {"task": task_obj}
        
    # PHASE 2: Results & Receipts
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        task_id = message.get("taskId")
        if task_id not in db_tasks or db_task_owners.get(task_id) != token:
            raise HTTPException(status_code=404, detail="Task not found")
            
        async with task_locks[task_id]:
            task_obj = db_tasks[task_id]
            if task_obj["state"] != "TASK_STATE_INPUT_REQUIRED":
                raise HTTPException(status_code=409, detail="Task is not INPUT_REQUIRED")
            
            logging.info(f"\n--- GRADER RESULTS RECEIVED (Task: {task_id}) ---")
            
            executions = []
            prev_proposals = task_obj["artifacts"][0]["data"]["proposals"]
            prop_map = {p["actionId"]: p for p in prev_proposals}
            
            for res in data.get("results", []):
                if res.get("outcome") == "ACCEPTED":
                    p = prop_map.get(res.get("actionId"))
                    if not p or p["packageId"] != res.get("packageId") or p["action"] != res.get("action"):
                        raise HTTPException(status_code=400, detail="Mismatch")
                        
                    logging.info(f"[Action: {res.get('actionId')}] Accepted by grader. Saving receipt.")
                    
                    executions.append({
                        "packageId": p["packageId"],
                        "actionId": p["actionId"],
                        "action": p["action"],
                        "receiptNonce": res.get("receiptNonce"),
                        "facts": p["facts"],
                        "evidenceRefs": p["evidenceRefs"]
                    })
                    
            task_obj["history"].append(message)
            task_obj["state"] = "TASK_STATE_COMPLETED"
            task_obj["artifacts"].append({
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {"batchId": data.get("batchId"), "executions": executions}
            })
            
            db_tasks[task_id] = task_obj
            db_idempotency[idem_key] = {"hash": msg_hash, "task_id": task_id}
            
            save_state_for_examiner()
            return {"task": task_obj}

    raise HTTPException(status_code=400, detail="Unknown media type")

# --- Auxiliary Routes ---
@app.get("/a2a/tasks")
async def list_tasks(token: str = Depends(get_auth_token), _=Depends(enforce_a2a_headers)):
    return {"tasks": [task for tid, task in db_tasks.items() if db_task_owners.get(tid) == token]}

@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, token: str = Depends(get_auth_token), _=Depends(enforce_a2a_headers)):
    if task_id not in db_tasks or db_task_owners.get(task_id) != token:
        raise HTTPException(status_code=404, detail="Not found")
    return {"task": db_tasks[task_id]}

@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, token: str = Depends(get_auth_token), _=Depends(enforce_a2a_headers)):
    if task_id not in db_tasks or db_task_owners.get(task_id) != token:
        raise HTTPException(status_code=404, detail="Not found")
        
    async with task_locks[task_id]:
        task = db_tasks[task_id]
        if task["state"] in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
            raise HTTPException(status_code=409, detail="Already finalized")
            
        logging.info(f"--- TASK CANCELED: {task_id} ---")
        task["state"] = "TASK_STATE_CANCELED"
        save_state_for_examiner()
        return {"task": task}
# --- Examiner Debug Route ---
@app.get("/debug/state")
async def view_offline_state():
    """Allows the offline examiner to view your saved tasks directly in their browser."""
    try:
        with open("offline_grader_state.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"message": "No state saved yet. Run a test first."}