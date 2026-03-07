import threading
import requests
import time
import random

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"
DIM = 128

def worker_add(worker_id):
    headers = {"X-API-Key": API_KEY}
    for i in range(10):
        data = {
            "vectors": [[random.random() for _ in range(DIM)]],
            "ids": [worker_id * 100 + i],
            "metadata": [{"worker": worker_id, "iter": i}]
        }
        response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
        if response.status_code != 200:
            print(f"Worker {worker_id} ADD failed: {response.status_code} {response.text}")
        time.sleep(0.01)

def worker_search(worker_id):
    headers = {"X-API-Key": API_KEY}
    for i in range(10):
        data = {
            "queries": [[random.random() for _ in range(DIM)]],
            "k": 2
        }
        response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
        if response.status_code != 200:
            print(f"Worker {worker_id} SEARCH failed: {response.status_code} {response.text}")
        time.sleep(0.01)

def test_concurrency():
    headers = {"X-API-Key": API_KEY}
    # Init DB with larger M and ef_construction
    requests.post(f"{BASE_URL}/init", json={"dim": DIM, "storage_path": "test_concurrent.h5", "M": 32, "ef_construction": 200}, headers=headers)

    # Add some initial vectors
    data = {
        "vectors": [[random.random() for _ in range(DIM)] for _ in range(100)],
        "ids": list(range(1000, 1100))
    }
    requests.post(f"{BASE_URL}/add", json=data, headers=headers)

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker_add, args=(i,))
        threads.append(t)
        t.start()

    for i in range(5, 10):
        t = threading.Thread(target=worker_search, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Concurrency test finished")

if __name__ == "__main__":
    test_concurrency()
