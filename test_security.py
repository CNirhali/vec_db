import os
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "supersecretkey"

def test_metrics_protected():
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"/metrics status (no key): {response.status_code}")
    assert response.status_code == 401

    response = requests.get(f"{BASE_URL}/metrics", headers={"X-API-Key": API_KEY})
    print(f"/metrics status (with key): {response.status_code}")
    assert response.status_code == 200

def test_status_protected():
    response = requests.get(f"{BASE_URL}/status")
    print(f"/status status (no key): {response.status_code}")
    assert response.status_code == 401

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

def test_dos_protection_limits():
    headers = {"X-API-Key": API_KEY}

    # Test dimension limit (10,000)
    data = {"dim": 10001, "storage_path": "test_too_big.h5"}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with dim=10001: {response.status_code}")
    assert response.status_code == 422

    # Test batch size limit for Add (10,000)
    # We don't need to send huge vectors, just many small ones
    data = {"vectors": [[0.1]] * 10001}
    # Init first with small dim
    requests.post(f"{BASE_URL}/init", json={"dim": 1, "storage_path": "test_batch.h5"}, headers=headers)
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Add with 10001 vectors: {response.status_code}")
    assert response.status_code == 422

    # Test batch size limit for Search (10,000)
    data = {"queries": [[0.1]] * 10001, "k": 5}
    response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
    print(f"Search with 10001 queries: {response.status_code}")
    assert response.status_code == 422

    # Test batch size limit for Delete (10,000)
    data = {"ids": list(range(10001))}
    response = requests.post(f"{BASE_URL}/delete", json=data, headers=headers)
    print(f"Delete with 10001 ids: {response.status_code}")
    assert response.status_code == 422

    # Test batch size limit for Update (10,000)
    data = {"ids": list(range(10001)), "vectors": [[0.1]] * 10001}
    response = requests.post(f"{BASE_URL}/update", json=data, headers=headers)
    print(f"Update with 10001 vectors: {response.status_code}")
    assert response.status_code == 422

def test_dos_hnsw_parameters():
    headers = {"X-API-Key": API_KEY}

    # Test ef_construction limit (1,000)
    data = {"dim": 128, "storage_path": "test_ef.h5", "ef_construction": 1001}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with ef_construction=1001: {response.status_code}")
    assert response.status_code == 422

    # Test M limit (128)
    data = {"dim": 128, "storage_path": "test_m.h5", "M": 129}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with M=129: {response.status_code}")
    assert response.status_code == 422

def test_new_security_protections():
    headers = {"X-API-Key": API_KEY}

    # Test ef_search limit (1,000)
    data = {"dim": 128, "storage_path": "test_efs.h5", "ef_search": 1001}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with ef_search=1001: {response.status_code}")
    assert response.status_code == 422

    # Test storage_path extension
    data = {"dim": 128, "storage_path": "test_no_ext"}
    response = requests.post(f"{BASE_URL}/init", json=data, headers=headers)
    print(f"Init with test_no_ext: {response.status_code}")
    assert response.status_code == 422
    assert "storage_path must have a .h5 or .hdf5 extension" in response.text

    # Test vector dimension limit (10,000)
    # Init first
    requests.post(f"{BASE_URL}/init", json={"dim": 128, "storage_path": "test_dim.h5"}, headers=headers)

    data = {"vectors": [[0.1] * 10001]}
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Add with vector dim=10001: {response.status_code}")
    assert response.status_code == 422
    assert "Vector dimension exceeds limit of 10000" in response.text

    data = {"ids": [1], "vectors": [[0.1] * 10001]}
    response = requests.post(f"{BASE_URL}/update", json=data, headers=headers)
    print(f"Update with vector dim=10001: {response.status_code}")
    assert response.status_code == 422
    assert "Vector dimension exceeds limit of 10000" in response.text

    # Test query dimension limit (10,000)
    data = {"queries": [[0.1] * 10001], "k": 1}
    response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
    print(f"Search with query dim=10001: {response.status_code}")
    assert response.status_code == 422
    assert "Query dimension exceeds limit of 10000" in response.text

def test_nan_inf_protection():
    import json
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    requests.post(f"{BASE_URL}/init", json={"dim": 2, "storage_path": "security_test.h5"}, headers=headers)

    # Test NaN in Add
    data = {"vectors": [[float('nan'), 0.1]], "ids": [1]}
    payload = json.dumps(data)
    response = requests.post(f"{BASE_URL}/add", data=payload, headers=headers)
    print(f"Add NaN: {response.status_code}")
    assert response.status_code == 400
    assert "non-finite values" in response.text

    # Test Inf in Search
    data = {"queries": [[float('inf'), 0.1]], "k": 1}
    payload = json.dumps(data)
    response = requests.post(f"{BASE_URL}/search", data=payload, headers=headers)
    print(f"Search Inf: {response.status_code}")
    assert response.status_code == 400
    assert "non-finite values" in response.text

def test_delete_non_existent_id():
    # Security: Ensure deleting a non-existent ID does not crash the server (DoS)
    headers = {"X-API-Key": API_KEY}
    requests.post(f"{BASE_URL}/init", json={"dim": 128, "storage_path": "security_test.h5"}, headers=headers)
    response = requests.post(f"{BASE_URL}/delete", json={"ids": [999999]}, headers={"X-API-Key": API_KEY})
    print(f"Delete non-existent ID response: {response.status_code}")
    assert response.status_code == 200

def test_metadata_limits():
    headers = {"X-API-Key": API_KEY}
    requests.post(f"{BASE_URL}/init", json={"dim": 2, "storage_path": "meta_test.h5"}, headers=headers)

    # Test key count limit in Add (100)
    large_metadata = {f"key_{i}": i for i in range(101)}
    data = {
        "vectors": [[1.0, 1.0]],
        "metadata": [large_metadata]
    }
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Add with 101 keys: {response.status_code}")
    assert response.status_code == 422
    assert "Metadata entry exceeds limit of 100 keys" in response.text

    # Test size limit in Add (10 KB)
    huge_val = "x" * 11000
    large_metadata = {"key": huge_val}
    data = {
        "vectors": [[1.0, 1.0]],
        "metadata": [large_metadata]
    }
    response = requests.post(f"{BASE_URL}/add", json=data, headers=headers)
    print(f"Add with >10KB metadata: {response.status_code}")
    assert response.status_code == 422
    assert "Metadata entry size exceeds limit of 10 KB" in response.text

    # Test key count limit in Search (100)
    large_filter = {f"key_{i}": i for i in range(101)}
    data = {
        "queries": [[1.0, 1.0]],
        "filter_metadata": large_filter
    }
    response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
    print(f"Search with 101 filter keys: {response.status_code}")
    assert response.status_code == 422
    assert "Filter metadata exceeds limit of 100 keys" in response.text

    # Test size limit in Search (10 KB)
    large_filter = {"key": huge_val}
    data = {
        "queries": [[1.0, 1.0]],
        "filter_metadata": large_filter
    }
    response = requests.post(f"{BASE_URL}/search", json=data, headers=headers)
    print(f"Search with >10KB filter: {response.status_code}")
    assert response.status_code == 422
    assert "Filter metadata size exceeds limit of 10 KB" in response.text

def test_uninitialized_search_dos():
    # Security: Ensure searching an uninitialized DB does not crash the server (DoS)
    headers = {"X-API-Key": API_KEY}
    # Use a unique path to ensure it's fresh
    requests.post(f"{BASE_URL}/init", json={"dim": 128, "storage_path": "uninit_test.h5"}, headers=headers)

    search_data = {
        "queries": [[0.1] * 128],
        "k": 5
    }
    response = requests.post(f"{BASE_URL}/search", json=search_data, headers=headers)
    print(f"Search uninitialized DB response: {response.status_code}")
    assert response.status_code == 200
    assert response.json()["labels"] == [[]]

if __name__ == "__main__":
    test_metrics_protected()
    test_status_protected()
    test_path_traversal()
    test_dos_k_parameter()
    test_dos_protection_limits()
    test_dos_hnsw_parameters()
    test_new_security_protections()
    test_nan_inf_protection()
    test_delete_non_existent_id()
    test_metadata_limits()
    test_uninitialized_search_dos()
    print("ALL TESTS PASSED")
