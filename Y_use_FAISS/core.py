import numpy as np
from .index import HNSWIndex
from .storage import DiskStorage

class VectorDB:
    """
    High-performance, disk-persistent vector database.
    """
    def __init__(self, dim, storage_path, ef_construction=400, M=32, ef_search=128):
        """Initialize the DB with vector dimension, storage path, and HNSW parameters."""
        self.dim = dim
        self.index = HNSWIndex(dim, ef_construction=ef_construction, M=M, ef_search=ef_search)
        self.storage = DiskStorage(storage_path, dim)

    def add(self, vectors, ids=None, metadata=None):
        """Add vectors (numpy array), optional ids, and optional metadata to the DB."""
        self.index.add(vectors, ids)
        self.storage.save_vectors(vectors, ids, metadata)

    def search(self, queries, k=10, filter_metadata=None):
        """Search for k nearest neighbors for each query vector, optionally filter by metadata."""
        labels, distances = self.index.search(queries, k)
        if filter_metadata is not None:
            # Load all metadata and filter results
            _, all_ids, all_metadata = self.storage.load_vectors()
            id_to_meta = {int(i): m for i, m in zip(all_ids, all_metadata)}
            filtered_labels = []
            filtered_distances = []
            for i, row in enumerate(labels):
                filtered = [(l, distances[i][j]) for j, l in enumerate(row) if id_to_meta.get(int(l), {}) == filter_metadata or (filter_metadata.items() <= id_to_meta.get(int(l), {}).items())]
                if filtered:
                    filtered_labels.append([l for l, _ in filtered])
                    filtered_distances.append([d for _, d in filtered])
                else:
                    filtered_labels.append([])
                    filtered_distances.append([])
            return filtered_labels, filtered_distances
        return labels, distances

    def save(self):
        """Persist index and storage to disk."""
        self.index.save()
        self.storage.save()

    def load(self):
        """Load index and storage from disk."""
        self.index.load()
        self.storage.load()

    def delete(self, ids):
        """Delete vectors by ID from the DB."""
        self.index.delete(ids)
        self.storage.delete_vectors(ids)

    def update(self, ids, vectors):
        """Update vectors by ID (delete old, add new)."""
        self.delete(ids)
        self.add(vectors, ids) 