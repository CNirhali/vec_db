import h5py
import numpy as np
import json
import os

class DiskStorage:
    """
    Disk-backed storage for vectors using HDF5, with metadata support.
    """
    def __init__(self, path, dim):
        self.path = path
        self.dim = dim
        self.vectors = None
        self.ids = None
        self.metadata = None

    def get_dim(self):
        """Get the dimension of existing vectors in the storage file."""
        if not os.path.exists(self.path):
            return None
        with h5py.File(self.path, 'r') as f:
            if 'vectors' in f:
                return f['vectors'].shape[1]
        return None

    def save_vectors(self, vectors, ids=None, metadata=None):
        """Save vectors, optional ids, and optional metadata to disk."""
        with h5py.File(self.path, 'a') as f:
            if 'vectors' not in f:
                f.create_dataset('vectors', data=vectors, maxshape=(None, self.dim), chunks=True, compression='gzip')
            else:
                dset = f['vectors']
                dset.resize((dset.shape[0] + vectors.shape[0], self.dim))
                dset[-vectors.shape[0]:] = vectors
            if ids is not None:
                if 'ids' not in f:
                    f.create_dataset('ids', data=ids, maxshape=(None,), chunks=True, compression='gzip')
                else:
                    idset = f['ids']
                    idset.resize((idset.shape[0] + len(ids),))
                    idset[-len(ids):] = ids
            n_new = vectors.shape[0]
            if metadata is not None:
                # Store metadata as a JSON string array using variable-length strings to prevent truncation
                dt = h5py.string_dtype(encoding='utf-8')
                if 'metadata' not in f:
                    # Security: Backfill empty metadata for existing vectors to maintain alignment
                    n_existing = f['vectors'].shape[0] - n_new
                    full_metadata = [{}] * n_existing + list(metadata)
                    meta_json = [json.dumps(m) for m in full_metadata]
                    f.create_dataset('metadata', data=meta_json, maxshape=(None,), dtype=dt, chunks=True, compression='gzip')
                else:
                    mset = f['metadata']
                    mset.resize((mset.shape[0] + n_new,))
                    meta_json = [json.dumps(m) for m in metadata]
                    mset[-n_new:] = meta_json
            elif 'metadata' in f:
                # Security: Metadata not provided but dataset exists, append empty dicts to maintain alignment
                mset = f['metadata']
                mset.resize((mset.shape[0] + n_new,))
                empty_meta = [json.dumps({})] * n_new
                mset[-n_new:] = empty_meta

    def load_ids(self):
        """Load all IDs from disk."""
        if not os.path.exists(self.path):
            return np.array([], dtype=np.int64)
        with h5py.File(self.path, 'r') as f:
            if 'ids' not in f:
                return np.array([], dtype=np.int64)
            return f['ids'][:]

    def load_vectors(self):
        """Load all vectors, ids, and metadata from disk."""
        if not os.path.exists(self.path):
            return np.zeros((0, self.dim)), np.array([], dtype=np.int64), None
        with h5py.File(self.path, 'r') as f:
            if 'vectors' not in f:
                return np.zeros((0, self.dim)), np.array([], dtype=np.int64), None
            vectors = f['vectors'][:]
            ids = f['ids'][:] if 'ids' in f else np.array([], dtype=np.int64)
            if 'metadata' in f:
                meta_json = f['metadata'][:]
                # Robustly decode metadata, handling both bytes and str
                metadata = [json.loads(m.decode('utf-8') if isinstance(m, bytes) else m) for m in meta_json]
            else:
                metadata = None
        return vectors, ids, metadata

    def save_metadata(self, metadata):
        """Save metadata array to disk (overwrite)."""
        with h5py.File(self.path, 'a') as f:
            if 'metadata' in f:
                del f['metadata']
            dt = h5py.string_dtype(encoding='utf-8')
            meta_json = [json.dumps(m) for m in metadata]
            f.create_dataset('metadata', data=meta_json, maxshape=(None,), dtype=dt, chunks=True, compression='gzip')

    def load_metadata(self, indices=None):
        """Load metadata array from disk, optionally filtered by indices."""
        with h5py.File(self.path, 'r') as f:
            if 'metadata' in f:
                if indices is not None:
                    # Security: Load only specific metadata entries to prevent OOM
                    meta_json = f['metadata'][indices]
                else:
                    meta_json = f['metadata'][:]
                # Robustly decode metadata, handling both bytes and str
                return [json.loads(m.decode('utf-8') if isinstance(m, bytes) else m) for m in meta_json]
            else:
                return None

    def save(self):
        """Placeholder for additional save logic if needed."""
        pass

    def load(self):
        """Placeholder for additional load logic if needed."""
        pass

    def delete_vectors(self, ids):
        """Delete vectors and ids from disk by ID."""
        vectors, all_ids, metadata = self.load_vectors()
        if len(all_ids) == 0:
            return

        # Security: Safeguard against inconsistent state (IDs vs vectors length mismatch)
        if len(all_ids) != len(vectors):
            # If state is already corrupted, we can't safely use boolean masking.
            # We fail gracefully instead of crashing the server.
            raise RuntimeError(f"Storage state corrupted: {len(all_ids)} IDs for {len(vectors)} vectors. Deletion aborted to prevent further corruption.")

        mask = ~np.isin(all_ids, ids)
        # If no changes are needed, return early
        if np.all(mask):
            return
        vectors = vectors[mask]
        all_ids = all_ids[mask]
        if metadata is not None:
            metadata = [m for i, m in enumerate(metadata) if mask[i]]
        with h5py.File(self.path, 'a') as f:
            if 'vectors' in f:
                del f['vectors']
            if 'ids' in f:
                del f['ids']
            if 'metadata' in f:
                del f['metadata']
            f.create_dataset('vectors', data=vectors, maxshape=(None, self.dim), chunks=True, compression='gzip')
            f.create_dataset('ids', data=all_ids, maxshape=(None,), chunks=True, compression='gzip')
            if metadata is not None:
                dt = h5py.string_dtype(encoding='utf-8')
                meta_json = [json.dumps(m) for m in metadata]
                f.create_dataset('metadata', data=meta_json, maxshape=(None,), dtype=dt, chunks=True, compression='gzip')