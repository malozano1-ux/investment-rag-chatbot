# send_single.py
# Send a single WhatsApp template message to one recipient.
# Useful for testing templates before running a full campaign.
#
# Required env vars: BOT_BASE_URL, WHATSAPP_RECIPIENT, TEMPLATE_NAME, TEMPLATE_LANG
# (see .env.example)

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BOT_BASE_URL", "https://your-deployment-url.com").rstrip("/")
ENDPOINT = "/admin/send-template"

WA_ID = (os.getenv("WHATSAPP_RECIPIENT") or "YOUR_TEST_PHONE_NUMBER").strip().lstrip("+")
TEMPLATE_NAME = (os.getenv("TEMPLATE_NAME") or "your_template_name").strip()
LANG = (os.getenv("TEMPLATE_LANG") or "en").strip()

payload = {
    "wa_id": WA_ID,
    "name": TEMPLATE_NAME,
    "lang": LANG,
    "vars": {"customer_name": "Test User"},
}

url = f"{BASE_URL}{ENDPOINT}"

resp = requests.post(url, json=payload, timeout=30)
print("POST", url)
print("Status:", resp.status_code)
print("Response:", resp.text)
resp.raise_for_status()

data = resp.json()
meta_err = (data.get("resp") or {}).get("error")
if meta_err:
    print("\n❌ Meta returned an error:")
    print(meta_err)
    sys.exit(1)

print("\n✅ Sent successfully.")
