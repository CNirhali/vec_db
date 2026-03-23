import requests

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_metadata_filtering():
    headers = {"X-API-Key": API_KEY}

    # Init DB
    requests.post(f"{BASE_URL}/init", json={"dim": 2, "storage_path": "filter_test.h5"}, headers=headers)

    # Add data with metadata
    data = {
        "vectors": [[0.1, 0.1], [0.2, 0.2]],
        "metadata": [{"label": "cat"}, {"label": "dog"}]
    }
    requests.post(f"{BASE_URL}/add", json=data, headers=headers)

    # Search with filter
    search_data = {
        "queries": [[0.1, 0.1]],
        "k": 2,
        "filter_metadata": {"label": "cat"}
    }
    response = requests.post(f"{BASE_URL}/search", json=search_data, headers=headers)
    print(f"Filter response status: {response.status_code}")
    if response.status_code == 200:
        print(f"Filter response: {response.json()}")
    else:
        print(f"Filter response error: {response.text}")

if __name__ == "__main__":
    test_metadata_filtering()
