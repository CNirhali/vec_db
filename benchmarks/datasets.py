import numpy as np
import os
import urllib.request
import zipfile

def load_sift1m(data_dir=".cache/sift1m"):
    """Download and load SIFT1M base vectors and queries. Returns (base, queries)."""
    os.makedirs(data_dir, exist_ok=True)
    base_path = os.path.join(data_dir, "sift_base.fvecs")
    query_path = os.path.join(data_dir, "sift_query.fvecs")
    if not os.path.exists(base_path) or not os.path.exists(query_path):
        print("[INFO] Downloading SIFT1M dataset...")
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        local_tar = os.path.join(data_dir, "sift.tar.gz")
        urllib.request.urlretrieve(url, local_tar)
        import tarfile
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=data_dir)
    def read_fvecs(path):
        with open(path, 'rb') as f:
            dim = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(0)
            data = np.fromfile(f, dtype=np.float32)
            data = data.reshape(-1, dim + 1)
            return data[:, 1:]
    base = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    return base, queries

def load_glove(dim=100, data_dir=".cache/glove"):
    """Download and load GloVe vectors of given dimension. Returns numpy array."""
    os.makedirs(data_dir, exist_ok=True)
    glove_file = os.path.join(data_dir, f"glove.6B.{dim}d.txt")
    if not os.path.exists(glove_file):
        print("[INFO] Downloading GloVe dataset...")
        url = f"http://nlp.stanford.edu/data/glove.6B.zip"
        local_zip = os.path.join(data_dir, "glove.6B.zip")
        urllib.request.urlretrieve(url, local_zip)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    vectors = []
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == dim + 1:
                vectors.append([float(x) for x in parts[1:]])
    return np.array(vectors, dtype=np.float32)

def generate_synthetic(num_vectors=1000000, dim=128, seed=42):
    """Generate synthetic dataset for benchmarking."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(num_vectors, dim)).astype(np.float32)
    return vectors 