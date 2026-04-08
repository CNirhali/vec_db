import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_save_rate_limit():
    headers = {"X-API-Key": API_KEY}
    # Init first
    requests.post(f"{BASE_URL}/init", json={"dim": 1, "storage_path": "save_test.h5"}, headers=headers)

    print("Testing rate limiting on /save (10/minute)...")
    for i in range(12):
        response = requests.post(f"{BASE_URL}/save", headers=headers)
        if response.status_code == 429:
            print(f"Rate limit hit at request {i+1}")
            return
    print("Rate limit NOT hit after 12 requests")
    assert False, "Rate limit should have been hit"

if __name__ == "__main__":
    test_save_rate_limit()
