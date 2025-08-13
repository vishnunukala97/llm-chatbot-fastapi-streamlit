import os, json, time, random, urllib.request
from urllib.error import HTTPError, URLError

key   = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
if not key:
    raise SystemExit("GEMINI_API_KEY not set. Load .env or install python-dotenv.")

url  = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
body = {"contents":[{"role":"user","parts":[{"text":"Say: Hello from Gemini (one short line)."}]}]}

def call():
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
        return data["candidates"][0]["content"]["parts"][0].get("text","")

transient_codes = {429, 500, 502, 503, 504}
for attempt in range(1, 7):  # up to 6 tries
    try:
        text = call()
        print("OK:", text)
        break
    except HTTPError as e:
        msg = ""
        try:
            msg = e.read().decode()[:200]
        except Exception:
            pass
        print(f"HTTPError {e.code} on attempt {attempt}: {msg}")
        if e.code not in transient_codes:
            raise
    except URLError as e:
        print(f"URLError on attempt {attempt}: {e}")
    except Exception as e:
        print(f"Error on attempt {attempt}: {e}")

    # exponential backoff with a little jitter
    sleep_s = min(2 ** attempt, 30) + random.uniform(0, 0.5)
    print(f"Retrying in {sleep_s:.1f}s...")
    time.sleep(sleep_s)
else:
    raise SystemExit("Failed after retries.")
