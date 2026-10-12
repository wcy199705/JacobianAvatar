import numpy as np
import collections
from sklearn.cluster import KMeans
import numpy as np
import collections
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation # Axis-angle <-> Rotation Matrix conversion
from pytorch3d.transforms import matrix_to_rotation_6d,  axis_angle_to_matrix
import torch
# helper utilities
class RandomSampler:
    """
    Simple sampler that randomly shuffles data indices every epoch.
    Naive version of ClusteredBalancedSampler with the clustering and balanced-sampling logic removed.

    Usage:
    1. sampler = RandomSampler(num_samples=..., batch_size=...) # call once before training
    2. for epoch in range(num_epochs):
    3.     sampler.reshuffle(seed=epoch) # reshuffle the order every epoch
    4.     for batch_indices in sampler: # batch training
    5.         ...
    """
    def __init__(self, num_samples, batch_size, seed=None):
        """
        [Lightweight constructor] Initialize the sampler.

        Args:
            num_samples (int): number of samples in the whole dataset
            batch_size (int): size of each batch
            seed (int, optional): initial seed for reproducibility. Defaults to None.
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.base_seed = seed

        # Create all data indices from 0 to num_samples-1.
        self.indices = np.arange(num_samples)
        
        # Shuffle once for the first epoch.
        self.reshuffle(seed=self.base_seed)

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Call every epoch to reshuffle the data order.
        """
        current_seed = self.base_seed if seed is None else seed
        if current_seed is not None:
            # Set the seed for reproducibility.
            np.random.seed(current_seed)
        
        # Shuffle the index array in place.
        np.random.shuffle(self.indices)


    def __iter__(self):
        """Initialize state for iteration."""
        self.n = 0
        return self

    def __next__(self):
        """Return the next batch."""
        if self.n < len(self):
            start_idx = self.n * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = self.indices[start_idx:end_idx]
            self.n += 1
            return batch
        else:
            # Stop the loop when there are no more batches to return.
            raise StopIteration
        

class PairedRandomSampler:
    """
    Sampler that randomly shuffles data indices organized as (idx, idx+1) pairs every epoch.
    
    Key features:
    - Two adjacent indices are always sampled together.
    - `batch_size` means the number of 'pairs' returned. 
      Therefore the actual returned batch size is `2 * batch_size`. (The last batch may be smaller.)

    Usage:
    1. sampler = PairedRandomSampler(num_samples=..., batch_size=...) # call once before training
    2. for epoch in range(num_epochs):
    3.     sampler.reshuffle(seed=epoch) # reshuffle the order every epoch
    4.     for batch_indices in sampler: # batch training
    5.         # batch_indices has the form [idx1, idx1+1, idx2, idx2+1, ...].
    6.         ...
    """
    def __init__(self, num_samples, batch_size, seed=None):
        """
        [Lightweight constructor] Initialize the sampler.

        Args:
            num_samples (int): number of samples in the whole dataset.
            batch_size (int): number of 'pairs' in each batch.
            seed (int, optional): initial seed for reproducibility. Defaults to None.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 to form pairs.")

        self.num_total_samples = num_samples
        self.pair_batch_size = batch_size  # number of 'pairs' per batch
        self.base_seed = seed

        # Create all valid indices that can be the start of a pair.
        # e.g. if num_samples=100, valid start indices are 0 to 98.
        self.valid_start_indices = np.arange(num_samples - 1)
        self.num_pairs = len(self.valid_start_indices)
        
        # Shuffle once for the first epoch.
        self.reshuffle(seed=self.base_seed)

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Reshuffle the order of pairs every epoch.
        """
        current_seed = self.base_seed if seed is None else seed
        if current_seed is not None:
            np.random.seed(current_seed)
        
        # Shuffle the valid start-index array in place.
        np.random.shuffle(self.valid_start_indices)


    def __iter__(self):
        """Initialize state for iteration."""
        self.n = 0
        return self

    def __next__(self):
        """Return the next batch made of pairs."""
        if self.n < len(self):
            # 1. Get the 'pair start indices' for this batch.
            start = self.n * self.pair_batch_size
            end = start + self.pair_batch_size
            batch_starts = self.valid_start_indices[start:end]

            # 2. Build an (idx, idx+1) pair for each start index.
            # e.g. if batch_starts is [5, 98, 42],
            #     batch_pairs becomes [[5, 6], [98, 99], [42, 43]].
            batch_pairs = np.stack([batch_starts, batch_starts + 1], axis=1)

            # 3. Flatten into a 1D array for the final batch.
            # Result: [5, 6, 98, 99, 42, 43]
            
            self.n += 1
            return batch_pairs
        else:
            # Stop the loop when there are no more batches to return.
            raise StopIteration

class TripleRandomSampler:
    """
    Sampler that randomly shuffles data indices organized as (idx-1, idx, idx+1) groups every epoch.
    
    Key features:
    - Two adjacent indices are always sampled together.
    - `batch_size` means the number of 'pairs' returned. 
      Therefore the actual returned batch size is `2 * batch_size`. (The last batch may be smaller.)

    Usage:
    1. sampler = PairedRandomSampler(num_samples=..., batch_size=...) # call once before training
    2. for epoch in range(num_epochs):
    3.     sampler.reshuffle(seed=epoch) # reshuffle the order every epoch
    4.     for batch_indices in sampler: # batch training
    5.         # batch_indices has the form [idx1, idx1+1, idx2, idx2+1, ...].
    6.         ...
    """
    def __init__(self, num_samples, batch_size, seed=None):
        """
        [Lightweight constructor] Initialize the sampler.

        Args:
            num_samples (int): number of samples in the whole dataset.
            batch_size (int): number of 'pairs' in each batch.
            seed (int, optional): initial seed for reproducibility. Defaults to None.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 to form pairs.")

        self.num_total_samples = num_samples
        self.pair_batch_size = batch_size  # number of 'pairs' per batch
        self.base_seed = seed

        # Create all valid indices that can be the start of a pair.
        # e.g. if num_samples=100, valid start indices are 1 to 98.
        self.valid_start_indices = np.arange(1, num_samples - 1)
        self.num_pairs = len(self.valid_start_indices)
        
        # Shuffle once for the first epoch.
        self.reshuffle(seed=self.base_seed)

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Reshuffle the order of pairs every epoch.
        """
        current_seed = self.base_seed if seed is None else seed
        if current_seed is not None:
            np.random.seed(current_seed)
        
        # Shuffle the valid start-index array in place.
        np.random.shuffle(self.valid_start_indices)


    def __iter__(self):
        """Initialize state for iteration."""
        self.n = 0
        return self

    def __next__(self):
        """Return the next batch made of pairs."""
        if self.n < len(self):
            # 1. Get the 'pair start indices' for this batch.
            start = self.n * self.pair_batch_size
            end = start + self.pair_batch_size
            batch_starts = self.valid_start_indices[start:end]

            # 2. Build an (idx, idx+1) pair for each start index.
            # e.g. if batch_starts is [5, 98, 42],
            #     batch_pairs becomes [[5, 6], [98, 99], [42, 43]].
            batch_pairs = np.stack([batch_starts-1, batch_starts, batch_starts + 1], axis=1)

            # 3. Flatten into a 1D array for the final batch.
            # Result: [5, 6, 98, 99, 42, 43]
            
            self.n += 1
            return batch_pairs
        else:
            # Stop the loop when there are no more batches to return.
            raise StopIteration

class DualRandomSampler:
    """
    Sampler that randomly shuffles data indices organized as (idx-1, idx, idx+1) groups every epoch.
    
    Key features:
    - Two adjacent indices are always sampled together.
    - `batch_size` means the number of 'pairs' returned. 
      Therefore the actual returned batch size is `2 * batch_size`. (The last batch may be smaller.)

    Usage:
    1. sampler = PairedRandomSampler(num_samples=..., batch_size=...) # call once before training
    2. for epoch in range(num_epochs):
    3.     sampler.reshuffle(seed=epoch) # reshuffle the order every epoch
    4.     for batch_indices in sampler: # batch training
    5.         # batch_indices has the form [idx1, idx1+1, idx2, idx2+1, ...].
    6.         ...
    """
    def __init__(self, num_samples, batch_size, seed=None):
        """
        [Lightweight constructor] Initialize the sampler.

        Args:
            num_samples (int): number of samples in the whole dataset.
            batch_size (int): number of 'pairs' in each batch.
            seed (int, optional): initial seed for reproducibility. Defaults to None.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 to form pairs.")

        self.num_total_samples = num_samples
        self.pair_batch_size = batch_size  # number of 'pairs' per batch
        self.base_seed = seed

        # Create all valid indices that can be the start of a pair.
        # e.g. if num_samples=100, valid start indices are 1 to 98.
        self.valid_start_indices = np.arange(0, num_samples - 1)
        self.num_pairs = len(self.valid_start_indices)
        
        # Shuffle once for the first epoch.
        self.reshuffle(seed=self.base_seed)

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Reshuffle the order of pairs every epoch.
        """
        current_seed = self.base_seed if seed is None else seed
        if current_seed is not None:
            np.random.seed(current_seed)
        
        # Shuffle the valid start-index array in place.
        np.random.shuffle(self.valid_start_indices)


    def __iter__(self):
        """Initialize state for iteration."""
        self.n = 0
        return self

    def __next__(self):
        """Return the next batch made of pairs."""
        if self.n < len(self):
            # 1. Get the 'pair start indices' for this batch.
            start = self.n * self.pair_batch_size
            end = start + self.pair_batch_size
            batch_starts = self.valid_start_indices[start:end]

            # 2. Build an (idx, idx+1) pair for each start index.
            # e.g. if batch_starts is [5, 98, 42],
            #     batch_pairs becomes [[5, 6], [98, 99], [42, 43]].
            batch_pairs = np.stack([batch_starts, batch_starts + 1], axis=1)

            # 3. Flatten into a 1D array for the final batch.
            # Result: [5, 6, 98, 99, 42, 43]
            
            self.n += 1
            return batch_pairs
        else:
            # Stop the loop when there are no more batches to return.
            raise StopIteration


class ClusteredBalancedSampler:
    """
    Combines expensive clustering and lightweight per-epoch sampling logic in a single class.
    Usage:
    1. sampler = ClusteredBalancedSampler.initialize(...) # call once before training
    2. for epoch in ...:
    3.     sampler.reshuffle(seed=epoch) # reshuffle the order every epoch
    4.     for batch_indices in sampler: # batch training
    5.         ...
    """
    @classmethod
    def initialize(cls, global_orientations, num_groups, batch_size, seed=None):
        """
        [Expensive initialization method] Call only once before training starts.
        Adjusts axis-angle vectors and then clusters them with K-Means.
        """
        print("Starting initial clustering... (this runs only once)")
        if seed is not None:
            np.random.seed(seed)

        # 1. Adjust axis-angle vectors (handle rotational symmetry)
        print(" - Adjusting axis-angle vectors...")
        #angles = np.linalg.norm(global_orientations, axis=1)
        

        # Restrict angles to [0, pi] and flip negative axes to positive
        adjusted_orientations = global_orientations.copy()
        #axisangle to rot matrix -> rotmatrix -> axis_angle
        adjusted_orientations = axis_angle_to_matrix(torch.tensor(adjusted_orientations))
        adjusted_orientations = matrix_to_rotation_6d(adjusted_orientations).numpy()
        #mask = angles > np.pi
        # adjusted_orientations[mask] = -adjusted_orientations[mask] * (2 * np.pi - angles[mask, None]) / angles[mask, None]
        # Handle small or zero angles
        #mask = angles > 0
        #adjusted_orientations[~mask] = np.zeros_like(adjusted_orientations[~mask])

        # 2. K-Means clustering
        print(f" - K-Means (k={num_groups}) clustering...")
        kmeans = KMeans(
            n_clusters=num_groups,
            random_state=seed,
            init='k-means++',
            n_init=10
        )
        kmeans.fit(adjusted_orientations)
        cluster_labels = kmeans.labels_
        print("Initial clustering done.")
        #chekc clusters.
        return cls(cluster_labels, batch_size, seed)

    def __init__(self, cluster_labels, batch_size, seed=None):
        """
        [Lightweight constructor] Initialized with precomputed cluster labels.
        Usually called via initialize() rather than directly by the user.
        """
        self.cluster_labels = cluster_labels
        self.batch_size = batch_size
        self.base_seed = seed
        self.groups = self._group_indices()
        self.max_len = max(len(g) for g in self.groups.values()) if self.groups else 0
        self.reshuffle(seed)

    def _group_indices(self):
        """Group and store indices by label."""
        groups = collections.defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            groups[label].append(idx)
        return groups

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Call every epoch to reshuffle the data order.
        Clustering is not performed again.
        """
        current_seed = self.base_seed if seed is None else seed
        if current_seed is not None:
            np.random.seed(current_seed)
        resampled_groups = []
        for group_label in sorted(self.groups.keys()):
            group_indices = self.groups[group_label]
            resampled_indices = np.random.choice(group_indices, size=self.max_len, replace=True)
            resampled_groups.append(resampled_indices)
        interleaved_indices = []
        for i in range(self.max_len):
            for group in resampled_groups:
                interleaved_indices.append(group[i])
        self.interleaved_indices = interleaved_indices
        self.num_batches = (len(self.interleaved_indices) + self.batch_size - 1) // self.batch_size


    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batches:
            start_idx = self.n * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = self.interleaved_indices[start_idx:end_idx]
            self.n += 1
            return batch
        else:
            raise StopIteration
        

class ClusteredPairedSampler:
    """
    Combines the logic of ClusteredBalancedSampler and the Triple/Dual samplers.
    
    1. Select K-Means-clustered 'center' indices. (ClusteredBalancedSampler)
    2. Return the selected 'center' index together with its neighbors (idx-1, idx, idx+1).
    
    Boundary handling:
    - If the 'center' index is 0, returns (0, 0, 1). (triple mode)
    - If the 'center' index is (N-1), returns (N-2, N-1, N-1). (triple mode)

    Usage:
    1. sampler = ClusteredPairedSampler.initialize(
                   global_orientations=..., 
                   num_groups=..., 
                   batch_size=..., 
                   mode='triple' # choose 'triple' or 'dual'
               )
    2. for epoch in ...:
    3.     sampler.reshuffle(seed=epoch)
    4.     for batch_indices in sampler:
    5.         # with mode='triple', batch_indices.shape = (batch_size, 3)
    6.         # with mode='dual', batch_indices.shape = (batch_size, 2)
    7.         ...
    """
    
    @classmethod
    def initialize(cls, global_orientations, num_groups, batch_size, mode='triple', seed=None):
        """
        [Expensive initialization method] 
        Internally initializes a ClusteredBalancedSampler.
        
        Args:
            global_orientations (np.array): (N, D) array to be clustered.
            num_groups (int): k for K-Means.
            batch_size (int): number of 'pairs' or 'triplets' returned at once.
            mode (str): 'triple' (idx-1, idx, idx+1) or 'dual' (idx, idx+1).
            seed (int, optional): seed for reproducibility.
        """
        # 1. First, initialize a ClusteredBalancedSampler and delegate the 'center' index generation to it.
        balanced_sampler = ClusteredBalancedSampler.initialize(
            global_orientations=global_orientations,
            num_groups=num_groups,
            batch_size=batch_size,
            seed=seed
        )
        
        num_total_samples = len(global_orientations)
        
        # 2. Call the lightweight constructor of ClusteredPairedSampler.
        return cls(balanced_sampler, num_total_samples, mode)

    def __init__(self, balanced_sampler, num_total_samples, mode='triple'):
        """
        [Lightweight constructor] 
        Takes a pre-built ClusteredBalancedSampler instance.
        Usually called via initialize() rather than directly by the user.
        """
        self.balanced_sampler = balanced_sampler
        self.num_total_samples = num_total_samples
        
        if mode not in ['dual', 'triple']:
            raise ValueError("mode must be 'dual' or 'triple'.")
        if mode == 'triple' and self.num_total_samples < 3:
             print(f"Warning: 'triple' mode but the total number of samples is {self.num_total_samples}. Boundary clamping may occur frequently.")
        if mode == 'dual' and self.num_total_samples < 2:
             raise ValueError("'dual' mode requires at least 2 samples.")
        self.mode = mode

    def reshuffle(self, seed=None):
        """
        [Lightweight shuffle method] Reshuffle the inner sampler's order every epoch.
        """
        self.balanced_sampler.reshuffle(seed)


    def __iter__(self):
        """Reset the inner sampler for iteration."""
        # Call __iter__ of the inner sampler (balanced_sampler) 
        # to reset it to self.n = 0.
        self.balanced_sampler.__iter__() 
        return self

    def __next__(self):
            """
            [Revised logic]
            Return the next batch made of 'pairs' or 'triplets'.
            Clips 'center_indices' so that the (idx-1, idx, idx+1) relation
            is always preserved, even at the boundaries.
            """
            # 1. Get a batch of 'center' indices from the inner sampler.
            center_indices = next(self.balanced_sampler)
            center_indices = np.array(center_indices)

            # 2. Clip the 'center' indices to the valid range for the mode.
            if self.mode == 'triple':
                # 'center' can only be from 1 to (N-2).
                # 0 -> 1 and (N-1) -> (N-2).
                # (if N=3, min=1 and max=1, so all indices are clipped to 1)
                valid_min = 1
                valid_max = self.num_total_samples - 2
                
                adjusted_centers = np.clip(center_indices, valid_min, valid_max)
                
                # 3. Compute neighbors around the clipped center.
                # Thanks to the N>=3 guarantee in __init__, this never goes out of range.
                prev_indices = adjusted_centers - 1
                next_indices = adjusted_centers + 1
                
                batch_groups = np.stack([prev_indices, adjusted_centers, next_indices], axis=1)

            else: # self.mode == 'dual'
                # 'center' can only be from 0 to (N-2).
                # (N-1) -> (N-2).
                # (if N=2, min=0 and max=0, so all indices are clipped to 0)
                valid_min = 0
                valid_max = self.num_total_samples - 2

                adjusted_centers = np.clip(center_indices, valid_min, valid_max)
                
                # 3. Compute neighbors around the clipped center.
                next_indices = adjusted_centers + 1
                
                batch_groups = np.stack([adjusted_centers, next_indices], axis=1)

            return batch_groups


class ClusteredPairedSampler_except_sample:
    """
    [Fully safe version]
    Samples from a dataset that never contains invalid indices, without retry loops.
    
    How it works:
    1. At initialization, invalid indices and the last frame (N-1) are removed from the candidates.
    2. The ClusteredBalancedSampler is built only from the remaining 'valid indices'.
    3. Sampled values are mapped back to original indices before returning.
    """
    
    @classmethod
    def initialize(cls, global_orientations, num_groups, batch_size, invalid_indices=None, seed=None):
        num_total_samples = len(global_orientations)
        
        # 1. Preprocess invalid indices (convert to set for faster lookup)
        if invalid_indices is not None:
            invalid_set = set(invalid_indices)
        else:
            invalid_set = set()
            
        # 2. Build a mask that keeps only valid indices
        # Condition A: must not be in invalid_indices
        # Condition B: the last frame (N-1) has no 'Next', so it cannot be 'Current'
        valid_mask = []
        for i in range(num_total_samples):
            if i == num_total_samples - 1: # exclude the last frame
                valid_mask.append(False)
            elif i in invalid_set:         # exclude user-specified invalid indices
                valid_mask.append(False)
            else:
                valid_mask.append(True)
        
        valid_mask = np.array(valid_mask, dtype=bool)
        
        # 3. Build the mapping table (Filtered Index -> Original Index)
        # e.g. original [0, 1, 2(X), 3] -> valid_map [0, 1, 3]
        # if the sampler picks 2 -> returns valid_map[2] = 3
        valid_map = np.arange(num_total_samples)[valid_mask]
        
        if len(valid_map) < 2:
            raise ValueError(f"Too few valid frames. (Valid: {len(valid_map)})")

        # 4. Extract only the 'valid data' for clustering
        filtered_orientations = global_orientations[valid_mask]
        
        # 5. Initialize the inner sampler (cluster using valid data only)
        balanced_sampler = ClusteredBalancedSampler.initialize(
            global_orientations=filtered_orientations, # pass filtered data, not the full set
            num_groups=num_groups,
            batch_size=batch_size,
            seed=seed
        )
        
        return cls(balanced_sampler, valid_map)

    def __init__(self, balanced_sampler, valid_map):
        """
        Args:
            balanced_sampler: inner sampler built only from valid data
            valid_map (np.array): table mapping inner-sampler indices to original indices
        """
        self.balanced_sampler = balanced_sampler
        self.valid_map = valid_map
        self.sampler_iter = None

    def reshuffle(self, seed=None):
        self.balanced_sampler.reshuffle(seed)


    def __iter__(self):
        # Create and store the inner sampler's iterator
        self.sampler_iter = iter(self.balanced_sampler)
        return self

    def __next__(self):
        # 1. Check the iterator
        if self.sampler_iter is None:
             self.sampler_iter = iter(self.balanced_sampler)
             
        try:
            # 2. Get indices from the inner sampler
            # These indices are relative to 'filtered_orientations' (0 ~ len(valid)-1)
            # so they can never be invalid or out of range.
            local_indices = next(self.sampler_iter)
        except StopIteration:
            raise StopIteration
            
        local_indices = np.array(local_indices)

        # 3. [Key] Map local indices to original dataset indices
        # e.g. local 0 -> original 0, local 2 -> original 3 (if 2 was invalid)
        current_indices = self.valid_map[local_indices]
        
        # 4. Compute the Next frame
        # current already excludes the last frame and invalid indices,
        # so the +1 value always exists and is within the dataset range.
        next_indices = current_indices + 1
        
        # 5. Return the result
        batch_pairs = np.stack([current_indices, next_indices], axis=1)

        return batch_pairs
    
import numpy as np

import numpy as np

class TripleClusteredPairedSampler_w_except:
    """
    [Fully safe - Triple version (supports Custom Link + Custom Curr)]
    Samples only cases where all three frames (Prev, Curr, Next) are safe, without retry loops.
    
    Options:
        If custom_prev/curr/next_indices are given, 
        the final indices are returned via those mapping tables instead of simple arithmetic.
    """
    
    @classmethod
    def initialize(cls, global_orientations, num_groups, batch_size, 
                   invalid_indices=None, seed=None,
                   custom_prev_indices=None, custom_curr_indices=None, custom_next_indices=None):
        """
        Args:
            custom_prev_indices (np.array, optional): array of length N. Prev value of the i-th sample.
            custom_curr_indices (np.array, optional): array of length N. Curr value of the i-th sample. (defaults to 0..N-1)
            custom_next_indices (np.array, optional): array of length N. Next value of the i-th sample.
        """
        num_total_samples = len(global_orientations)
        curr_global_orient = global_orientations[custom_curr_indices]
        num_total_sub_samples = len(curr_global_orient)

        # -----------------------------------------------------------
        # 0. Define relations (Prev, Next)
        # -----------------------------------------------------------
        if custom_prev_indices is not None and custom_next_indices is not None:
            # [Case A] Use user-specified relations
            if len(custom_prev_indices) != num_total_sub_samples or len(custom_next_indices) != num_total_sub_samples:
                raise ValueError("Custom Prev/Next indices arrays must have the same length as global_orientations.")
            
            all_prev_map = np.array(custom_prev_indices)
            all_next_map = np.array(custom_next_indices)
            use_custom_link = True
        else:
            # [Case B] Use default consecutive relations (i-1, i+1)
            all_prev_map = np.arange(num_total_sub_samples) - 1
            all_next_map = np.arange(num_total_sub_samples) + 1
            use_custom_link = False

        # -----------------------------------------------------------
        # 0-1. Define the current index mapping (Curr) - **NEW**
        # -----------------------------------------------------------
        if custom_curr_indices is not None:
            if len(custom_curr_indices) != num_total_sub_samples:
                raise ValueError("Custom Curr indices array must have the same length as global_orientations.")
            all_curr_map = np.array(custom_curr_indices)
        else:
            # Default if not given (0, 1, 2, ... N-1)
            all_curr_map = np.arange(num_total_sub_samples)

        # -----------------------------------------------------------
        # 1. Preprocess invalid indices
        # -----------------------------------------------------------
        if invalid_indices is not None:
            invalid_set = set(invalid_indices)
        else:
            invalid_set = set()
            
        # -----------------------------------------------------------
        # 2. Build a mask that keeps only valid centers (rows)
        # -----------------------------------------------------------
        valid_mask = []
        for i in range(num_total_sub_samples):
            # i is the dataset 'Row Index'.
            # Check whether the Prev/Next 'values' this row points to are valid.
            prev_val = all_prev_map[i]
            next_val = all_next_map[i]
                
            # [Condition 1] Range and existence check
            if use_custom_link:
                # Custom mode: a value of -1 (or negative) means no neighbor
                if prev_val < 0 or next_val < 0:
                    valid_mask.append(False)
                    continue
                # Index range check (must not exceed the dataset range)
                if prev_val >= num_total_samples or next_val >= num_total_samples:
                    valid_mask.append(False)
                    continue
            else:
                # Default mode: exclude the first and last frames
                if i == 0 or i == num_total_samples - 1:
                    valid_mask.append(False)
                    continue
            
            # [Condition 2] The Curr row itself must not be invalid
            if i in invalid_set:
                valid_mask.append(False)
                continue
                
            # [Condition 3] The Prev target must not be invalid
            #if prev_val in invalid_set:
            #    valid_mask.append(False)
            #    continue
                
            # [Condition 4] The Next target must not be invalid
            if next_val in invalid_set:
                valid_mask.append(False)
                continue
            
            # True if all conditions pass
            valid_mask.append(True)
        
        valid_mask = np.array(valid_mask, dtype=bool)
        
        # -----------------------------------------------------------
        # 3. Build the mapping table (Filtered Index -> Original Row Index)
        # -----------------------------------------------------------
        valid_map = np.arange(num_total_sub_samples)[valid_mask]
        
        if len(valid_map) < 1:
            raise ValueError(f"Cannot build any valid Triple frames.")

        # 4. Extract only the 'valid data' for clustering
        filtered_orientations = curr_global_orient[valid_mask]
        
        # 5. Initialize the inner sampler
        balanced_sampler = ClusteredBalancedSampler.initialize(
            global_orientations=filtered_orientations,
            num_groups=num_groups,
            batch_size=batch_size,
            seed=seed
        )
        
        # Pass all relation maps (prev, curr, next) to the instance
        return cls(balanced_sampler, valid_map, all_prev_map, all_curr_map, all_next_map)

    def __init__(self, balanced_sampler, valid_map, all_prev_map, all_curr_map, all_next_map):
        self.balanced_sampler = balanced_sampler
        self.valid_map = valid_map
        
        # Look-up Tables
        self.all_prev_map = all_prev_map
        self.all_curr_map = all_curr_map # **NEW**
        self.all_next_map = all_next_map
        
        self.sampler_iter = None

    def reshuffle(self, seed=None):
        self.balanced_sampler.reshuffle(seed)


    def __iter__(self):
        self.sampler_iter = iter(self.balanced_sampler)
        return self

    def __next__(self):
        if self.sampler_iter is None:
             self.sampler_iter = iter(self.balanced_sampler)
             
        try:
            # 1. Get (filtered) indices from the inner sampler
            local_indices = next(self.sampler_iter)
        except StopIteration:
            raise StopIteration
            
        local_indices = np.array(local_indices)

        # 2. Restore local indices to the original dataset 'Row Index'
        original_row_indices = self.valid_map[local_indices]
        
        # 3. Look up final return values (using the mapping tables)
        # Use the Row Index as the key to find the actual value to return.
        final_prev = self.all_prev_map[original_row_indices]
        final_curr = self.all_curr_map[original_row_indices] # **NEW: Curr is mapped too**
        final_next = self.all_next_map[original_row_indices]
        
        # 4. Return the result (Batch, 3) -> [Prev, Curr, Next]
        batch_triplets = np.stack([final_prev, final_curr, final_next], axis=1)

        return batch_triplets