import os
import json
import urllib.request
import urllib.error

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Key not found")
else:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"

    data = {
        "contents": [
            {"parts": [{"text": "Say hello"}]}
        ]
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
            print(result["candidates"][0]["content"]["parts"][0]["text"])
    except urllib.error.HTTPError as error:
        print("HTTP Error:", error.code)
        print(error.read().decode("utf-8")[:1000])