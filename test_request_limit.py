import requests
import json
import os

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_request_limit():
    # 151MB payload to exceed 150MB limit
    payload_size = 151 * 1024 * 1024

    # Using 'x' * payload_size to create a large string for the request body
    # Note: requests will automatically add Content-Length header
    data = "x" * payload_size

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Sending {len(data) / (1024*1024):.2f} MB request...")
    try:
        # We don't care about valid JSON here, the middleware should check Content-Length first
        response = requests.post(f"{BASE_URL}/add", data=data, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        assert response.status_code == 413
        assert "Payload Too Large" in response.text
    except Exception as e:
        print(f"Error during request: {e}")
        # If the server closes connection because it's too large, it might also be acceptable,
        # but our middleware should return 413.
        raise e

def test_metadata_list_limit():
    headers = {"X-API-Key": API_KEY}
    # Init first
    requests.post(f"{BASE_URL}/init", json={"dim": 2, "storage_path": "meta_limit_test.h5"}, headers=headers)

    # Test metadata list length limit (10,000)
    # We want 10,001 metadata entries
    data = {
        "vectors": [[0.1, 0.1]] * 10001,
        "metadata": [{}] * 10001
    }
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Metadata list size 10001 response: {response.status_code}")
    assert response.status_code == 422
    assert "List should have at most 10000 items" in response.text

if __name__ == "__main__":
    try:
        test_request_limit()
        test_metadata_list_limit()
        print("REQUEST LIMIT TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        exit(1)
