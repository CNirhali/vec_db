import requests
import numpy as np

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_search_output_dos():
    headers = {"X-API-Key": API_KEY}

    # Init DB
    dim = 1
    requests.post(f"{BASE_URL}/init", json={"dim": dim, "storage_path": "dos_test.h5"}, headers=headers)

    # Add some data
    requests.post(f"{BASE_URL}/add", json={"vectors": [[0.1]] * 100}, headers=headers)

    # Large search request that passes current validation but produces huge output
    # len(queries) = 10000, k = 1000
    # Total output elements = 10,000,000
    queries = [[0.1]] * 10000
    data = {
        "queries": queries,
        "k": 1000
    }

    print("Sending large search request...")
    try:
        response = requests.post(f"{BASE_URL}/search", json=data, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        # If it returns 200, it means the server spent a lot of resources
    except requests.exceptions.Timeout:
        print("Request timed out - potentially causing DoS")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_search_output_dos()
