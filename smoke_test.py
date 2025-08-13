import os, json, urllib.request
from urllib.error import HTTPError, URLError

key   = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not key:
    raise SystemExit("GEMINI_API_KEY not set. Load .env into PowerShell or use python-dotenv.")

url  = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
body = {"contents":[{"role":"user","parts":[{"text":"Say: Hello from Gemini (one short line)."}]}]}
req  = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        text = data["candidates"][0]["content"]["parts"][0].get("text","")
        print("OK:", text)
except HTTPError as e:
    print("HTTPError", e.code, e.read().decode()[:200])
except URLError as e:
    print("URLError:", e)
except Exception as e:
    print("Error:", e)
