import h5py
import numpy as np
import json

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
            if metadata is not None:
                # Store metadata as a JSON string array
                meta_json = np.array([json.dumps(m) for m in metadata], dtype='S')
                if 'metadata' not in f:
                    f.create_dataset('metadata', data=meta_json, maxshape=(None,), chunks=True, compression='gzip')
                else:
                    mset = f['metadata']
                    mset.resize((mset.shape[0] + len(meta_json),))
                    mset[-len(meta_json):] = meta_json

    def load_vectors(self):
        """Load all vectors, ids, and metadata from disk."""
        with h5py.File(self.path, 'r') as f:
            vectors = f['vectors'][:]
            ids = f['ids'][:] if 'ids' in f else None
            if 'metadata' in f:
                meta_json = f['metadata'][:]
                metadata = [json.loads(m.decode('utf-8')) for m in meta_json]
            else:
                metadata = None
        return vectors, ids, metadata

    def save_metadata(self, metadata):
        """Save metadata array to disk (overwrite)."""
        with h5py.File(self.path, 'a') as f:
            if 'metadata' in f:
                del f['metadata']
            meta_json = np.array([json.dumps(m) for m in metadata], dtype='S')
            f.create_dataset('metadata', data=meta_json, maxshape=(None,), chunks=True, compression='gzip')

    def load_metadata(self):
        """Load metadata array from disk."""
        with h5py.File(self.path, 'r') as f:
            if 'metadata' in f:
                meta_json = f['metadata'][:]
                return [json.loads(m.decode('utf-8')) for m in meta_json]
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
        if all_ids is None:
            return
        mask = ~np.isin(all_ids, ids)
        vectors = vectors[mask]
        all_ids = all_ids[mask]
        if metadata is not None:
            metadata = [m for i, m in enumerate(metadata) if mask[i]]
        with h5py.File(self.path, 'a') as f:
            del f['vectors']
            del f['ids']
            if 'metadata' in f:
                del f['metadata']
            f.create_dataset('vectors', data=vectors, maxshape=(None, self.dim), chunks=True, compression='gzip')
            f.create_dataset('ids', data=all_ids, maxshape=(None,), chunks=True, compression='gzip')
            if metadata is not None:
                meta_json = np.array([json.dumps(m) for m in metadata], dtype='S')
                f.create_dataset('metadata', data=meta_json, maxshape=(None,), chunks=True, compression='gzip') 