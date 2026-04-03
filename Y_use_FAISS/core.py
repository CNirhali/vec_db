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
        # Security: Maintain in-memory ID-to-position mapping for efficient O(1) collision checks and metadata filtering.
        # This prevents Denial-of-Service (DoS) by avoiding O(N) disk scans or large vector operations during search.
        self.id_to_pos = {}

        # Security: Derive index path from storage path and attempt recovery
        self.index_path = os.path.splitext(storage_path)[0] + ".bin"
        self._load_or_init_index()

    def _load_or_init_index(self):
        """Load index from disk or rebuild from storage if missing."""
        if os.path.exists(self.index_path):
            self.index.load(self.index_path)
            # Rebuild ID-to-position mapping from storage
            all_ids = self.storage.load_ids()
            self.id_to_pos = {int(v_id): i for i, v_id in enumerate(all_ids)}
        elif os.path.exists(self.storage.path):
            # Security: Rebuild index from storage if bin file is missing but data exists
            vectors, ids, _ = self.storage.load_vectors()
            if len(vectors) > 0:
                self.index.add(vectors, ids)
                self.id_to_pos = {int(v_id): i for i, v_id in enumerate(ids)}

    def add(self, vectors, ids=None, metadata=None):
        """Add vectors (numpy array), optional ids, and optional metadata to the DB."""
        with self.lock:
            if ids is not None:
                # Security: Prevent cross-tenant data leakage and index corruption by ensuring provided IDs are unique and do not already exist.
                # Use O(1) in-memory hash map for collision checks instead of I/O heavy storage lookups.
                if any(int(v_id) in self.id_to_pos for v_id in ids):
                    raise ValueError("One or more provided IDs already exist in the database. Use the /update endpoint to modify existing vectors.")

            ids = self.index.add(vectors, ids)
            self.storage.save_vectors(vectors, ids, metadata)

            # Update mapping with new vector positions
            current_count = len(self.id_to_pos)
            for i, v_id in enumerate(ids):
                self.id_to_pos[int(v_id)] = current_count + i

    def search(self, queries, k=10, filter_metadata=None):
        """Search for k nearest neighbors for each query vector, optionally filter by metadata."""
        # Index search in hnswlib is thread-safe for reading
        labels, distances = self.index.search(queries, k)
        if filter_metadata is not None:
            # Security: Optimize memory usage by only loading metadata for the ANN-returned results.
            # Using self.id_to_pos avoids O(N) disk scans or large vectorized operations on all IDs, mitigating a major DoS vector.
            with self.lock:
                # Map unique labels to their positional indices in storage
                unique_labels = np.unique(labels)
                valid_labels = [int(l) for l in unique_labels if int(l) in self.id_to_pos]

                if not valid_labels:
                    id_to_meta = {}
                else:
                    indices_to_load = [self.id_to_pos[l] for l in valid_labels]
                    loaded_metadata = self.storage.load_metadata(indices=indices_to_load)
                    if loaded_metadata is None:
                        id_to_meta = {}
                    else:
                        id_to_meta = {l: meta for l, meta in zip(valid_labels, loaded_metadata)}

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
            # Rebuild mapping after deletion since positions change
            all_ids = self.storage.load_ids()
            self.id_to_pos = {int(v_id): i for i, v_id in enumerate(all_ids)}

    def update(self, ids, vectors):
        """Update vectors by ID (delete old, add new)."""
        # Lock is handled within delete and add, but we wrap here too for atomicity
        with self.lock:
            self.index.delete(ids)
            self.storage.delete_vectors(ids)
            # Re-add with same IDs to ensure they are replaced in storage properly
            # and update the in-memory mapping.
            # We bypass self.add() here to avoid the uniqueness check on IDs we just deleted.
            added_ids = self.index.add(vectors, ids)
            self.storage.save_vectors(vectors, added_ids)
            # Rebuild mapping after update since positions shift during deletion/append
            all_ids = self.storage.load_ids()
            self.id_to_pos = {int(v_id): i for i, v_id in enumerate(all_ids)}