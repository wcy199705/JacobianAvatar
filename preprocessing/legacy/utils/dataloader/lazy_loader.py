"""
Lazy loading utilities for reducing RAM usage.
Images/masks/normals are loaded from disk on-demand instead of preloading all into memory.
"""
import os
import cv2
import numpy as np


class LazyImageList:
    """
    Drop-in replacement for a list of images.
    Stores file paths and loads images from disk on access.
    Supports: indexing, len(), iteration, list comprehension.

    Usage:
        lazy = LazyImageList(file_paths, preprocess_fn=some_fn)
        img = lazy[0]           # loads from disk
        imgs = [lazy[i] for i in idxs]  # loads subset
        stacked = np.stack([lazy[i] for i in idxs])  # works as expected
    """

    def __init__(self, file_paths, preprocess_fn=None):
        """
        Args:
            file_paths: list of image file paths
            preprocess_fn: optional function(np.ndarray) -> np.ndarray
                           applied after cv2.imread for each image
        """
        self.file_paths = file_paths
        self.preprocess_fn = preprocess_fn
        self._shape_cache = None  # cached (h, w, c) from first access

    def _load(self, idx):
        img = cv2.imread(self.file_paths[idx])
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        return img

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._load(idx)
        elif isinstance(idx, slice):
            return [self._load(i) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, (list, np.ndarray)):
            return [self._load(i) for i in idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self._load(i)

    @property
    def shape(self):
        """Return (N, H, W, C) shape info without loading all images."""
        if self._shape_cache is None:
            sample = self._load(0)
            self._shape_cache = sample.shape
        n = len(self)
        h, w = self._shape_cache[:2]
        c = self._shape_cache[2] if len(self._shape_cache) > 2 else 1
        return (n, h, w, c)

    def append(self, item):
        raise NotImplementedError("LazyImageList is read-only. Cannot append.")


class LazyMaskList(LazyImageList):
    """Lazy loader for grayscale masks. Returns uint8 (H, W) arrays."""

    def _load(self, idx):
        mask = cv2.imread(self.file_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.preprocess_fn is not None:
            mask = self.preprocess_fn(mask)
        return mask


class LazyNormalArray:
    """
    Lazy loader for normal maps stored as .npy files.
    Supports numpy-style indexing: normals[idxs] returns a stacked array.
    """

    def __init__(self, file_paths, preprocess_fn=None, dtype=np.float16):
        self.file_paths = file_paths
        self.preprocess_fn = preprocess_fn
        self.dtype = dtype
        self._shape_cache = None

    def _load(self, idx):
        normal = np.load(self.file_paths[idx])
        if self.preprocess_fn is not None:
            normal = self.preprocess_fn(normal)
        return normal.astype(self.dtype)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._load(idx)
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return np.stack([self._load(i) for i in indices])
        elif isinstance(idx, (list, np.ndarray)):
            return np.stack([self._load(int(i)) for i in idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self._load(i)

    @property
    def shape(self):
        if self._shape_cache is None:
            sample = self._load(0)
            self._shape_cache = sample.shape
        n = len(self)
        return (n, *self._shape_cache)
