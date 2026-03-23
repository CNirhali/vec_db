import numpy as np
from .index import HNSWIndex
from .storage import DiskStorage
import threading
import os

class VectorDB:
    """
    High-performance, disk-persistent vector database.
    """
    def __init__(self, dim, storage_path, ef_construction=400, M=32, ef_search=128):
        """Initialize the DB with vector dimension, storage path, and HNSW parameters."""
        self.dim = dim
        self.storage = DiskStorage(storage_path, dim)

        # Security: Verify that the provided dimension matches existing storage to prevent corruption
        existing_dim = self.storage.get_dim()
        if existing_dim is not None and existing_dim != dim:
            raise ValueError(f"Dimension mismatch: storage has dim {existing_dim}, but {dim} was requested.")

        self.index = HNSWIndex(dim, ef_construction=ef_construction, M=M, ef_search=ef_search)
        self.lock = threading.Lock()

        # Security: Derive index path from storage path and attempt recovery
        self.index_path = os.path.splitext(storage_path)[0] + ".bin"
        self._load_or_init_index()

    def _load_or_init_index(self):
        """Load index from disk or rebuild from storage if missing."""
        if os.path.exists(self.index_path):
            self.index.load(self.index_path)
        elif os.path.exists(self.storage.path):
            # Security: Rebuild index from storage if bin file is missing but data exists
            vectors, ids, _ = self.storage.load_vectors()
            if len(vectors) > 0:
                self.index.add(vectors, ids)

    def add(self, vectors, ids=None, metadata=None):
        """Add vectors (numpy array), optional ids, and optional metadata to the DB."""
        with self.lock:
            ids = self.index.add(vectors, ids)
            self.storage.save_vectors(vectors, ids, metadata)

    def search(self, queries, k=10, filter_metadata=None):
        """Search for k nearest neighbors for each query vector, optionally filter by metadata."""
        # Index search in hnswlib is thread-safe for reading
        labels, distances = self.index.search(queries, k)
        if filter_metadata is not None:
            # Load all metadata and filter results
            with self.lock:
                _, all_ids, all_metadata = self.storage.load_vectors()
            if all_metadata is None or all_ids is None:
                id_to_meta = {}
            else:
                id_to_meta = {int(i): m for i, m in zip(all_ids, all_metadata)}

            filtered_labels = []
            filtered_distances = []
            for i, row in enumerate(labels):
                # Security: metadata filtering with support for exact match or subset match
                # Use all() for safe comparison of potentially unhashable values (e.g. lists) in metadata
                filtered = [(l, distances[i][j]) for j, l in enumerate(row)
                            if id_to_meta.get(int(l), {}) == filter_metadata or
                            (isinstance(id_to_meta.get(int(l)), dict) and
                             all(id_to_meta.get(int(l), {}).get(key) == value for key, value in filter_metadata.items()))]
                if filtered:
                    filtered_labels.append([int(l) for l, _ in filtered])
                    filtered_distances.append([float(d) for _, d in filtered])
                else:
                    filtered_labels.append([])
                    filtered_distances.append([])
            # Return as numpy arrays for consistency with index.search results
            return np.array(filtered_labels, dtype=object), np.array(filtered_distances, dtype=object)
        return labels, distances

    def save(self):
        """Persist index and storage to disk."""
        with self.lock:
            # Security: Use the correct index path for persistence
            self.index.save(self.index_path)
            self.storage.save()

    def load(self):
        """Load index and storage from disk."""
        self.index.load()
        self.storage.load()

    def delete(self, ids):
        """Delete vectors by ID from the DB."""
        with self.lock:
            self.index.delete(ids)
            self.storage.delete_vectors(ids)

    def update(self, ids, vectors):
        """Update vectors by ID (delete old, add new)."""
        # Lock is handled within delete and add, but we wrap here too for atomicity
        with self.lock:
            self.index.delete(ids)
            ids = self.index.add(vectors, ids)
            self.storage.delete_vectors(ids)
            self.storage.save_vectors(vectors, ids)