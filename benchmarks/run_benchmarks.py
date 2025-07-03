import time
import numpy as np
import psutil
import gc
import tracemalloc
import cProfile
import pstats
from concurrent.futures import ThreadPoolExecutor
from Y_use_FAISS.core import VectorDB
from .datasets import generate_synthetic, load_sift1m, load_glove
import faiss
import os
import pandas as pd
# import pinecone

try:
    import pinecone
    pinecone_available = True
except ImportError:
    pinecone_available = False

# ---- Dataset selection ----
# Options: 'synthetic', 'sift1m', 'glove'
DATASET = 'synthetic'

# ---- Dataset loading ----
def get_dataset():
    if DATASET == 'synthetic':
        dim = 128
        num_vectors = 100000  # Increased for realistic test
        vectors = generate_synthetic(num_vectors, dim)
        queries = vectors[:100]
        return vectors, queries, dim
    elif DATASET == 'sift1m':
        base, queries = load_sift1m()
        base = base[:10000]
        queries = queries[:10]
        dim = base.shape[1]
        return base, queries, dim
    elif DATASET == 'glove':
        dim = 100
        vectors = load_glove(dim)
        vectors = vectors[:10000]
        queries = vectors[:10]
        return vectors, queries, dim
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

# ---- Benchmarking helpers ----
def compute_true_neighbors(vectors, queries, k):
    """Compute true nearest neighbors using brute-force L2 distance."""
    dists = np.linalg.norm(vectors[None, :, :] - queries[:, None, :], axis=2)
    true_neighbors = np.argsort(dists, axis=1)[:, :k]
    return true_neighbors

def recall_at_k(results, true_neighbors, k):
    """Compute recall@k for ANN search results."""
    retrieved = results[0]  # hnswlib returns (labels, distances)
    recall = np.mean([
        len(set(retrieved[i][:k]) & set(true_neighbors[i])) / k
        for i in range(len(true_neighbors))
    ])
    return recall

# ---- Memory helpers ----
def get_tracemalloc_mb():
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 ** 2, peak / 1024 ** 2

# ---- Benchmarks ----
def get_process_memory_mb():
    gc.collect()
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2

def benchmark_vectordb(vectors, queries, dim, k=10, ef_construction=200, M=16, ef_search=64, profile=False, memory_only=False):
    print(f"[INFO] [VectorDB] vectors shape = {vectors.shape}, queries shape = {queries.shape}")
    db = VectorDB(dim, 'test_vectors.h5', ef_construction=ef_construction, M=M, ef_search=ef_search)
    tracemalloc.start()
    gc.collect()
    t0 = time.time()
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    if memory_only:
        # Only add to index, skip disk writes for pure speed
        db.index.add(vectors)
    else:
        # Batch add with background disk write
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(db.storage.save_vectors, vectors)
            db.index.add(vectors)
            future.result()
    if profile:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats('cumtime').print_stats(10)
    t1 = time.time()
    vdb_results = db.search(queries, k)
    t2 = time.time()
    current_mb, peak_mb = get_tracemalloc_mb()
    tracemalloc.stop()
    print(f"[DEBUG] Add: total={t1-t0:.4f}s, Search: {t2-t1:.4f}s, Mem current: {current_mb:.2f}MB, peak: {peak_mb:.2f}MB")
    true_neighbors = compute_true_neighbors(vectors, queries, k)
    recall = recall_at_k(vdb_results, true_neighbors, k)
    print(f"[RESULT] VectorDB recall@{k}: {recall:.4f}")
    return {'method': f'VectorDB_e{ef_construction}_M{M}_s{ef_search}_memonly{memory_only}', 'add_time': t1-t0, 'search_time': t2-t1, 'memory': peak_mb, 'recall': recall}

def benchmark_faiss(vectors, queries, dim, k=10):
    print(f"[INFO] [FAISS] vectors shape = {vectors.shape}, queries shape = {queries.shape}")
    index = faiss.IndexFlatL2(dim)
    start_mem = psutil.Process().memory_info().rss / 1024**2
    t0 = time.time()
    index.add(vectors)
    t1 = time.time()
    D, I = index.search(queries, k)
    t2 = time.time()
    end_mem = psutil.Process().memory_info().rss / 1024**2
    print(f"[RESULT] FAISS: Add time: {t1-t0:.2f}s, Search time: {t2-t1:.2f}s, Memory: {end_mem-start_mem:.2f}MB")
    print(f"[INFO] [FAISS] Sample search result for first query (top-5 IDs): {I[0][:5]}")
    print("[INFO] [FAISS] Computing brute-force recall@k (should be 1.0 for IndexFlatL2)...")
    true_neighbors = compute_true_neighbors(vectors, queries, k)
    recall = np.mean([
        len(set(I[i][:k]) & set(true_neighbors[i])) / k
        for i in range(len(true_neighbors))
    ])
    print(f"[RESULT] FAISS recall@{k}: {recall:.4f}")

def benchmark_pinecone(vectors, queries, dim, k=10):
    if not pinecone_available:
        print("[WARN] Pinecone not installed. Skipping Pinecone benchmark.")
        return
    print(f"[INFO] [Pinecone] vectors shape = {vectors.shape}, queries shape = {queries.shape}")
    api_key = os.environ.get("PINECONE_API_KEY", "<YOUR_API_KEY>")
    environment = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
    index_name = "benchmark-index"
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dim, metric="euclidean")
    index = pinecone.Index(index_name)
    ids = [str(i) for i in range(len(vectors))]
    batch_size = 10000
    t0 = time.time()
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        index.upsert(vectors=[(id, vec.tolist()) for id, vec in zip(batch_ids, batch_vectors)])
    t1 = time.time()
    results = []
    t2 = time.time()
    for q in queries:
        query_result = index.query(queries=[q.tolist()], top_k=k, include_values=False)
        results.append([int(match['id']) for match in query_result['matches']])
    t3 = time.time()
    print(f"[RESULT] Pinecone: Add time: {t1-t0:.2f}s, Search time: {t3-t2:.2f}s")
    print(f"[INFO] [Pinecone] Sample search result for first query (top-5 IDs): {results[0][:5]}")
    print("[INFO] [Pinecone] Computing brute-force recall@k (this may take a while)...")
    true_neighbors = compute_true_neighbors(vectors, queries, k)
    recall = np.mean([
        len(set(results[i][:k]) & set(true_neighbors[i])) / k
        for i in range(len(true_neighbors))
    ])
    print(f"[RESULT] Pinecone recall@{k}: {recall:.4f}")
    # pinecone.delete_index(index_name)

# ---- Main ----
def main():
    vectors, queries, dim = get_dataset()
    k = 10
    results = []
    true_neighbors = compute_true_neighbors(vectors, queries, k)
    # Parameter sweep: ultra-fast, low, medium, high recall
    param_sets = [
        {'ef_construction': 40, 'M': 4, 'ef_search': 8, 'memory_only': True},   # ultra-fast, lowest recall
        {'ef_construction': 100, 'M': 8, 'ef_search': 32, 'memory_only': True}, # low recall, fast
        {'ef_construction': 200, 'M': 16, 'ef_search': 64, 'memory_only': True},# medium
        {'ef_construction': 400, 'M': 32, 'ef_search': 128, 'memory_only': False}, # high recall, disk
        {'ef_construction': 800, 'M': 64, 'ef_search': 256, 'memory_only': True}, # max recall, in-memory
    ]
    for params in param_sets:
        results.append(benchmark_vectordb(vectors, queries, dim, k, **params, profile=False))
    # FAISS baseline
    t0 = time.time()
    index = faiss.IndexFlatL2(dim)
    tracemalloc.start()
    gc.collect()
    index.add(vectors)
    t1 = time.time()
    D, I = index.search(queries, k)
    t2 = time.time()
    current_mb, peak_mb = get_tracemalloc_mb()
    tracemalloc.stop()
    recall = np.mean([
        len(set(I[i][:k]) & set(true_neighbors[i])) / k
        for i in range(len(true_neighbors))
    ])
    results.append({'method': 'FAISS', 'add_time': t1-t0, 'search_time': t2-t1, 'memory': peak_mb, 'recall': recall})
    # Pinecone (if available)
    if pinecone_available:
        api_key = os.environ.get("PINECONE_API_KEY", "<YOUR_API_KEY>")
        environment = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
        index_name = "benchmark-index"
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dim, metric="euclidean")
        index_p = pinecone.Index(index_name)
        ids = [str(i) for i in range(len(vectors))]
        batch_size = 10000
        t0 = time.time()
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            index_p.upsert(vectors=[(id, vec.tolist()) for id, vec in zip(batch_ids, batch_vectors)])
        t1 = time.time()
        results_p = []
        t2 = time.time()
        for q in queries:
            query_result = index_p.query(queries=[q.tolist()], top_k=k, include_values=False)
            results_p.append([int(match['id']) for match in query_result['matches']])
        t3 = time.time()
        pinecone_recall = np.mean([
            len(set(results_p[i][:k]) & set(true_neighbors[i])) / k
            for i in range(len(true_neighbors))
        ])
        results.append({'method': 'Pinecone', 'add_time': t1-t0, 'search_time': t3-t2, 'memory': None, 'recall': pinecone_recall})
    # Save results
    RESULTS_CSV = 'benchmarks/results/benchmark_results.csv'
    os.makedirs('benchmarks/results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[RESULT] Saved benchmark results to {RESULTS_CSV}")

if __name__ == "__main__":
    main() 