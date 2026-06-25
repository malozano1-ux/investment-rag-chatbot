import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BOT_BASE_URL", "https://consultoria-4.onrender.com").rstrip("/")
ENDPOINT = "/admin/send-template"

WA_ID = (os.getenv("WHATSAPP_RECIPIENT") or "573114476006").strip().lstrip("+")
TEMPLATE_NAME = (os.getenv("TEMPLATE_NAME") or "mensaje_reengage_funding").strip()
LANG = (os.getenv("TEMPLATE_LANG") or "es_CO").strip()

payload = {
    "wa_id": WA_ID,
    "name": TEMPLATE_NAME,
    "lang": LANG,
    # Only include vars if your backend actually uses them to build components
    "vars": {"customer_name": "Felipe Leonardo"},
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

print("\n✅ Sent via backend; should be logged by log_to_sheet().")
