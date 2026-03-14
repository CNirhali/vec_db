import hnswlib
import numpy as np

class HNSWIndex:
    """
    HNSW-based ANN index for high recall and speed.
    """
    def __init__(self, dim, space='l2', ef_construction=400, M=32, ef_search=128):
        self.dim = dim
        self.space = space
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.index = hnswlib.Index(space=space, dim=dim)
        self.initialized = False

    def add(self, vectors, ids=None):
        """Add vectors to the index."""
        if not self.initialized:
            # For simplicity in this demo, we use a large max_elements or allow resizing if supported.
            # hnswlib requires a predefined max_elements, let's use a reasonably large default
            # or ideally manage it dynamically.
            self.index.init_index(max_elements=100000, ef_construction=self.ef_construction, M=self.M)
            self.initialized = True
        else:
            # Check if we need to resize
            current_count = self.index.get_current_count()
            if current_count + len(vectors) > self.index.get_max_elements():
                self.index.resize_index(max(current_count + len(vectors), self.index.get_max_elements() * 2))

        if ids is None:
            ids = np.arange(self.index.get_current_count(), self.index.get_current_count() + len(vectors))
        self.index.add_items(vectors, ids)
        return ids

    def search(self, queries, k=10):
        """Search for k nearest neighbors for each query vector."""
        if not self.initialized:
            # Security: If index is not initialized, return empty results to prevent DoS crash (hnswlib knn_query on uninitialized index)
            num_queries = queries.shape[0]
            # hnswlib.knn_query returns a tuple of (labels, distances)
            # labels is typically np.int64, distances is np.float32
            return np.zeros((num_queries, 0), dtype=np.int64), np.zeros((num_queries, 0), dtype=np.float32)
        self.index.set_ef(self.ef_search)
        return self.index.knn_query(queries, k=k)

    def save(self, path='hnsw_index.bin'):
        """Save the index to disk."""
        if not self.initialized:
            # Security: Prevent crash if trying to save an uninitialized index
            return
        self.index.save_index(path)

    def load(self, path='hnsw_index.bin'):
        """Load the index from disk."""
        self.index.load_index(path, max_elements=0)
        self.initialized = True

    def delete(self, ids):
        """Delete vectors by ID from the index (mark as deleted)."""
        if not self.initialized:
            return
        if hasattr(self.index, 'mark_deleted'):
            for vector_id in ids:
                try:
                    self.index.mark_deleted(vector_id)
                except RuntimeError as e:
                    # Security: Ignore "Label not found" to prevent API crash/500 on non-existent IDs
                    if "Label not found" in str(e):
                        continue
                    raise e
        else:
            raise NotImplementedError("Delete not supported by this index type.") 