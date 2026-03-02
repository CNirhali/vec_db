import os
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_metrics_protected():
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"/metrics status (no key): {response.status_code}")
    assert response.status_code == 422

    response = requests.get(f"{BASE_URL}/metrics", headers={"X-API-Key": API_KEY})
    print(f"/metrics status (with key): {response.status_code}")
    assert response.status_code == 200

def test_status_protected():
    response = requests.get(f"{BASE_URL}/status")
    print(f"/status status (no key): {response.status_code}")
    assert response.status_code == 422

    response = requests.get(f"{BASE_URL}/status", headers={"X-API-Key": API_KEY})
    print(f"/status status (with key): {response.status_code}")
    assert response.status_code == 200

def test_path_traversal():
    # Try to init with an absolute path
    data = {
        "dim": 128,
        "storage_path": "/tmp/test.h5"
    }
    headers = {"X-API-Key": API_KEY}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with /tmp/test.h5: {response.status_code} {response.text}")
    assert response.status_code == 422
    assert "storage_path must be a relative path" in response.text

    # Try with ..
    data = {
        "dim": 128,
        "storage_path": "../test.h5"
    }
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with ../test.h5: {response.status_code} {response.text}")
    assert response.status_code == 422

def test_dos_k_parameter():
    # Try to search with a very large k
    data = {
        "queries": [[0.1] * 128],
        "k": 1000000
    }
    headers = {"X-API-Key": API_KEY}
    # Need to init first
    requests.post(f"{BASE_URL}/init", json={"dim": 128, "storage_path": "test.h5"}, headers=headers)
    response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
    print(f"Search with k=1000000: {response.status_code} {response.text}")
    assert response.status_code == 422
    assert "Input should be less than or equal to 1000" in response.text

if __name__ == "__main__":
    test_metrics_protected()
    test_status_protected()
    test_path_traversal()
    test_dos_k_parameter()
    print("ALL TESTS PASSED")
