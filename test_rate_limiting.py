import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_rate_limiting_init():
    print("Testing rate limiting on /init (5/minute)...")
    headers = {"X-API-Key": API_KEY}
    # Use a unique storage path for each call to avoid other errors,
    # though here we just care about status code.

    status_codes = []
    for i in range(10):
        data = {"dim": 128, "storage_path": f"rate_limit_{i}.h5"}
        response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
        status_codes.append(response.status_code)
        print(f"Request {i+1}: {response.status_code}")
        if response.status_code == 429:
            break

    # We expect some 200s (up to 5) and then 429
    assert 200 in status_codes
    assert 429 in status_codes
    print("Rate limiting on /init verified.")

def test_rate_limiting_status():
    print("Testing rate limiting on /status (200/minute)...")
    headers = {"X-API-Key": API_KEY}

    status_codes = []
    for i in range(250):
        response = requests.get(f"{BASE_URL}/status", headers=headers)
        status_codes.append(response.status_code)
        if response.status_code == 429:
            print(f"Rate limit hit at request {i+1}")
            break
        if (i+1) % 50 == 0:
            print(f"Completed {i+1} requests...")

    assert 200 in status_codes
    assert 429 in status_codes
    print("Rate limiting on /status verified.")

if __name__ == "__main__":
    # Wait for server to be ready
    time.sleep(2)
    try:
        test_rate_limiting_init()
        test_rate_limiting_status()
        print("ALL RATE LIMITING TESTS PASSED")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
