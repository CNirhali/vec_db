import json
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_deep_metadata():
    headers = {"X-API-Key": API_KEY}
    # Create a 10KB deeply nested object
    # Max size is 10240 bytes.
    # '{"a":' is 5 bytes. 1000 levels is 5000 + 1000 = 6000 bytes.
    nested = '{"a":' * 1000 + '1' + '}' * 1000
    print(f"Metadata size: {len(nested)}")

    # We need to init first
    requests.post(f"{BASE_URL}/init", json={"dim": 1, "storage_path": "repro.h5"}, headers=headers)

    # Try to add vector with this metadata
    # The validation should pass because it only checks size and key count
    data = {
        "vectors": [[0.1]],
        "metadata": [json.loads(nested)]
    }
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Add response: {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return

    # Now search for it.
    # We don't even need to match it, but we want to trigger the SafeJSONEncoder if it's used.
    # Wait, SafeJSONEncoder is used on the WHOLE response body.
    # But metadata is NOT in the response body of /search.

    # Wait, if we added it, it's in storage.
    # Is there ANY endpoint that returns metadata? No.

    # But wait! The validation itself uses json.dumps(entry).
    # Does json.dumps(entry) fail with RecursionError?
    try:
        json.dumps(json.loads(nested))
        print("json.dumps passed")
    except RecursionError:
        print("json.dumps hit RecursionError")

test_deep_metadata()
