import os
import sys
import time
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BOT_BASE_URL", "https://consultoria-4.onrender.com").rstrip("/")
ENDPOINT = "/admin/send-template"

TEMPLATE_NAME = (os.getenv("TPL_ACCOUNT_OPENED") or "mensaje_efectivo_rentable").strip()
LANG = (os.getenv("TPL_LANG") or "es_CO").strip()

CSV_PATH = os.getenv("CASH_CSV", "cash.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
DELAY_BETWEEN_BATCHES = float(os.getenv("DELAY_BETWEEN_BATCHES", "5"))
DELAY_BETWEEN_MESSAGES = float(os.getenv("DELAY_BETWEEN_MESSAGES", "0.5"))

url = f"{BASE_URL}{ENDPOINT}"

def clean_wa_id(x: str) -> str:
    return (x or "").strip().lstrip("+").replace(" ", "")

def load_contacts(path: str):
    contacts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wa_id = clean_wa_id(row.get("wa_id", ""))
            name = (row.get("customer_name", "") or "").strip()
            if not wa_id:
                continue
            if not name:
                name = "hola"
            contacts.append({"wa_id": wa_id, "customer_name": name})
    return contacts

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

contacts = load_contacts(CSV_PATH)
if not contacts:
    print(f"❌ No contacts found in {CSV_PATH}")
    sys.exit(1)

print(f"✅ Loaded {len(contacts)} contacts from {CSV_PATH}")
print("POST", url)

sent = 0
failed = 0

for batch_num, batch in enumerate(chunked(contacts, BATCH_SIZE), start=1):
    print(f"\n=== Batch {batch_num} ({len(batch)} contacts) ===")

    for c in batch:
        payload = {
            "wa_id": c["wa_id"],
            "name": TEMPLATE_NAME,
            "lang": LANG,
            "vars": {"customer_name": c["customer_name"]},
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            ok = 200 <= resp.status_code < 300

            if not ok:
                failed += 1
                print(f"❌ {c['wa_id']} ({c['customer_name']}): HTTP {resp.status_code} {resp.text[:200]}")
                continue

            data = resp.json()
            meta_err = (data.get("resp") or {}).get("error")
            if meta_err:
                failed += 1
                print(f"❌ {c['wa_id']} ({c['customer_name']}): Meta error: {meta_err}")
                continue

            sent += 1
            print(f"✅ {c['wa_id']} ({c['customer_name']})")

        except Exception as e:
            failed += 1
            print(f"❌ {c['wa_id']} ({c['customer_name']}): Exception: {e}")

        time.sleep(DELAY_BETWEEN_MESSAGES)

    print(f"Batch {batch_num} done. Sleeping {DELAY_BETWEEN_BATCHES}s...")
    time.sleep(DELAY_BETWEEN_BATCHES)

print(f"\nDONE ✅ Sent={sent} Failed={failed} Total={len(contacts)}")
