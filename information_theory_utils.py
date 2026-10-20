from collections import Counter
import math
from itertools import combinations
from typing import Sequence, Tuple, Iterable, Dict, Any, Optional, Union, List
import os

# Try to import h5py and numpy for file storage, but make it optional
try:
    import h5py
    import numpy as np
    from scipy.special import digamma, polygamma, gammaln
    from scipy.linalg import pinv
    HDF5_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    SCIPY_AVAILABLE = False
    print("Warning: h5py, numpy, and/or scipy not available. File storage and Paninski estimators will be disabled.")


class StreamingInfo:
    """    
    # -----------------------------------------------------------------------------
    # information_theory_utils.py
    #
    # A **stream-friendly** container for information-theoretic statistics on many
    # discrete random variables.
    #
    # • Class  : `StreamingInfo`
    # • Purpose: Maintain only *count tables* (no raw data) while a stream of
    #   samples arrives, so you can query in-flight estimates of
    #       – H(X)                          (entropy)
    #       – H(X,Y,…)                      (joint entropy)
    #       – H(Y | X)                      (conditional entropy)
    #       – I(X;Y)                        (mutual information)
    #   for arbitrary singletons, pairs, triples, … that **you choose up front**.
    #
    # • Key Idea
    #   Each tracked variable-combo gets its own `collections.Counter`:
    #       counts[("X","Y")]   -> {(x_i, y_i): n_ij, …}
    #   Updates are O(1) per sample; queries run the usual ∑_p log p formula over
    #   the stored counts.  Memory = O(Σ |alphabet_combo|), unaffected by sample
    #   count N (unless you set `store_samples=True`).
    #
    # • Typical use
    #       smi = StreamingInfo(
    #               variables       = ["X1","X2","X3","X4"],
    #               combos_to_track = [("X1","X2"), ("X3","X4"),
    #                                  ("X1","X2","X3","X4")],
    #               base            = 2,         # bits
    #               store_samples   = False,
    #               store_samples_to_file = True,
    #               sample_file_path = "samples.h5")
    #       smi.update({"X1":0,"X2":1,"X3":0,"X4":1})
    #       mi = smi.mutual_information(("X1","X2"), ("X3","X4"))
    #
    # • Guarantees/Limitations
    #   – Will raise `ValueError` if you query a combo that was not declared.
    #   – No bias-correction; add-α or NSB smoothing can be wired into the private
    #     `_entropy_from_counts` helper if needed.
    #
    # -----------------------------------------------------------------------------
    """
    def __init__(
        self,
        variables: Sequence[str],
        combos_to_track: Union[Sequence[Tuple[str, ...]], None] = None,
        base: float = 2.0,
        store_samples: bool = False,
        store_samples_to_file: bool = False,
        sample_file_path: Union[str, None] = None,
        compression: str = 'gzip',
        chunk_size: int = 10000,
    ):
        self.variables: Tuple[str, ...] = tuple(variables)
        self.base = base
        self.store_samples = store_samples
        self.store_samples_to_file = store_samples_to_file
        self.samples: Union[List[Dict[str, Any]], None] = [] if store_samples else None

        # File storage parameters
        self.sample_file_path = sample_file_path
        self.compression = compression
        self.chunk_size = chunk_size
        self._h5_file = None
        self._h5_datasets = {}
        self._sample_count_in_file = 0

        # Default: every singleton and every unordered pair
        if combos_to_track is None:
            singles = [(v,) for v in self.variables]
            pairs = list(combinations(self.variables, 2))
            combos_to_track = singles + pairs

        # Canonicalise and de‑duplicate
        self.combos = {self._canon(c) for c in combos_to_track}

        self.N = 0  # total number of samples ingested
        self.counts: Dict[Tuple[str, ...], Counter] = {
            combo: Counter() for combo in self.combos
        }

        # Setup file storage if requested
        if self.store_samples_to_file and self.sample_file_path:
            self._setup_file_storage()

    def _setup_file_storage(self):
        """Initialize HDF5 file for sample storage"""
        if not HDF5_AVAILABLE:
            print("Warning: Cannot store samples to file. h5py/numpy not available.")
            self.store_samples_to_file = False
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.sample_file_path), exist_ok=True)
            
            # Open HDF5 file in append mode
            self._h5_file = h5py.File(self.sample_file_path, 'a')
            
            # Create datasets for each variable if they don't exist
            for var in self.variables:
                if var not in self._h5_file:
                    # Create resizable dataset with chunking and compression
                    self._h5_datasets[var] = self._h5_file.create_dataset(
                        var,
                        shape=(0,),  # Start with size 0
                        maxshape=(None,),  # Unlimited growth
                        dtype=np.int32,  # Assuming integer samples
                        chunks=(self.chunk_size,),
                        compression=self.compression,
                        compression_opts=9  # Max compression for gzip
                    )
                else:
                    self._h5_datasets[var] = self._h5_file[var]
                    
            # Track current sample count in file
            if self.variables and self.variables[0] in self._h5_datasets:
                self._sample_count_in_file = self._h5_datasets[self.variables[0]].shape[0]
                
            print(f"File storage initialized: {self.sample_file_path}")
            if self._sample_count_in_file > 0:
                print(f"Existing samples in file: {self._sample_count_in_file}")
                
        except Exception as e:
            print(f"Warning: Failed to setup file storage: {e}")
            self.store_samples_to_file = False
            if self._h5_file is not None:
                self._h5_file.close()
                self._h5_file = None

    # ------------------------------------------------------------------ #
    # Helper utilities
    def _canon(self, vars_) -> Tuple[str, ...]:
        """Return a deterministic, sorted tuple of variable names."""
        if isinstance(vars_, str):
            return (vars_,)
        return tuple(sorted(vars_))

    def _log(self, x: float) -> float:
        return math.log(x, self.base)

    def _entropy_from_counts(self, counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * self._log(c / total) for c in counts.values())

    def _entropy_from_counts_paninski(self, counts: Counter, method='bub', k_max=15, display_flag=False) -> Tuple[float, float]:
        """
        Paninski entropy estimator (BUB or BAG).
        
        Returns:
        --------
        entropy : float
            Estimated entropy
        error_bound : float  
            Upper bound on RMS error
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Paninski estimators require scipy. Please install scipy to use this feature.")
            
        if len(counts) == 0:
            return 0.0, 0.0
            
        # Convert counts to histogram array
        # The Paninski estimator works with counts, not the specific outcome values
        # So we just need the count values, regardless of what the keys are
        histogram = np.array(list(counts.values()), dtype=int)
        
        if len(histogram) == 0:
            return 0.0, 0.0
            
        entropy, error_bound = self._compute_paninski_entropy(histogram, method, k_max, display_flag)
        
        # Convert from bits to specified base
        if self.base != 2:
            entropy = entropy * math.log(2) / math.log(self.base)
            error_bound = error_bound * math.log(2) / math.log(self.base)
            
        return entropy, error_bound

    def _get_counts(self, vars_) -> Counter:
        combo = self._canon(vars_)
        try:
            return self.counts[combo]
        except KeyError as exc:
            raise ValueError(
                f"Combination {combo} not tracked. "
                "Add it via `combos_to_track` when constructing StreamingInfo."
            ) from exc

    def _check_no_samples(self) -> bool:
        """Check if no samples have been added and print warning if so."""
        if self.N == 0:
            print("Warning: No samples have been added to StreamingInfo. Returning 0.")
            return True
        return False

    # ------------------------------------------------------------------ #
    # Ingestion
    def update(self, sample: Dict[str, Any]) -> None:
        """
        Update with one observation.

        Parameters
        ----------
        sample : dict
            Mapping {var_name: realised_symbol}. **Must** contain every variable
            specified at construction time.
        """
        missing = set(self.variables) - sample.keys()
        if missing:
            raise KeyError(f"Sample missing variables: {missing}")

        self.N += 1
        for combo in self.combos:
            key = tuple(sample[v] for v in combo)
            self.counts[combo][key] += 1

        if self.store_samples:
            self.samples.append({v: sample[v] for v in self.variables})

        if self.store_samples_to_file and self._h5_file is not None:
            # Append sample to HDF5 file
            for var_name, value in sample.items():
                self._h5_datasets[var_name].resize(self._sample_count_in_file + 1, axis=0)
                self._h5_datasets[var_name][self._sample_count_in_file] = value
            self._sample_count_in_file += 1

    def update_batch(self, samples: Iterable[Dict[str, Any]]) -> None:
        """Update with an iterable of sample dicts."""
        for s in samples:
            self.update(s)

    def flush_to_file(self):
        """Force write buffered data to disk"""
        if self._h5_file is not None:
            self._h5_file.flush()

    def close_file(self):
        """Close HDF5 file and cleanup resources"""
        if self._h5_file is not None:
            self._h5_file.close()
            print(f"Closed sample file: {self.sample_file_path}")

    def get_file_sample_count(self) -> int:
        """Get the number of samples stored in file"""
        return self._sample_count_in_file

    def load_samples_from_file(self) -> Dict[str, list]:
        """Load all samples from the HDF5 file (use with caution for large files)"""
        if not self.store_samples_to_file or self._h5_file is None:
            return {}
        
        samples = {}

        for var in self.variables:
            if var in self._h5_datasets:
                samples[var] = self._h5_datasets[var][:].tolist()
        
        return samples

    def __del__(self):
        """Cleanup on deletion"""
        self.close_file()

    # ------------------------------------------------------------------ #
    # Public information measures
    def entropy(self, vars_) -> float:
        """Shannon entropy H(vars_)."""
        if self._check_no_samples():
            return 0.0
        return self._entropy_from_counts(self._get_counts(vars_))

    # Alias for readability
    joint_entropy = entropy

    def entropy_paninski(self, vars_, method='bub', k_max=15, display_flag=False, domain_size_for_normalization=None) -> Tuple[float, float]:
        """
        Shannon entropy H(vars_) using Paninski's BUB or BAG estimator.
        
        Parameters
        ----------
        vars_ : str | Sequence[str]
            Variables to compute entropy for
        method : str
            'bub' for BUB estimator, 'bag' for BAG estimator  
        k_max : int
            Degree of freedom parameter for BUB
        display_flag : bool
            Whether to display diagnostic information
        domain_size_for_normalization : int, optional
            If provided, normalizes entropy by log(domain_size) to get values in [0,1]
            
        Returns
        -------
        entropy : float
            Estimated entropy (normalized if domain_size_for_normalization provided)
        error_bound : float
            Upper bound on RMS error (normalized if domain_size_for_normalization provided)
        """
        if self._check_no_samples():
            return 0.0, 0.0
        
        entropy, error_bound = self._entropy_from_counts_paninski(self._get_counts(vars_), method, k_max, display_flag)
        
        if domain_size_for_normalization is not None:
            normalization_factor = self._log(domain_size_for_normalization)
            entropy = entropy / normalization_factor
            error_bound = error_bound / normalization_factor
            
        return entropy, error_bound
    
    # Alias for readability  
    def joint_entropy_paninski(self, vars_, method='bub', k_max=15, display_flag=False, domain_size_for_normalization=None) -> Tuple[float, float]:
        """Joint entropy using Paninski estimator."""
        return self.entropy_paninski(vars_, method, k_max, display_flag, domain_size_for_normalization)

    def conditional_entropy(self, y_vars, given_x_vars) -> float:
        """
        H(Y | X) = H(X, Y) − H(X)

        Parameters
        ----------
        y_vars : str | Sequence[str]
        given_x_vars : str | Sequence[str]
        """
        if self._check_no_samples():
            return 0.0
        y = self._canon(y_vars)
        x = self._canon(given_x_vars)
        return self.joint_entropy(x + y) - self.entropy(x)

    def conditional_entropy_paninski(self, y_vars, given_x_vars, method='bub', k_max=15, display_flag=False, normalize=False) -> Tuple[float, float]:
        """
        H(Y | X) = H(X, Y) − H(X) using Paninski estimator
        
        Parameters
        ----------
        y_vars : str | Sequence[str]
        given_x_vars : str | Sequence[str]
        method : str
            'bub' for BUB estimator, 'bag' for BAG estimator
        k_max : int
            Degree of freedom parameter for BUB
        display_flag : bool
            Whether to display diagnostic information
        normalize : bool
            When True, normalizes conditional entropy by H(Y) to measure 
            the fraction of remaining uncertainty in Y that could not be explained by X
            
        Returns
        -------
        conditional_entropy : float
            Estimated conditional entropy (normalized if normalize=True)
        error_bound : float
            Combined error bound (approximate, normalized if normalize=True)
        """
        if self._check_no_samples():
            return 0.0, 0.0
            
        y = self._canon(y_vars)
        x = self._canon(given_x_vars)
        
        # Compute all required entropies in one batch to potentially reuse computations
        if normalize:
            # Need H(X,Y), H(X), and H(Y) for normalization
            h_xy, err_xy = self.joint_entropy_paninski(x + y, method, k_max, display_flag)
            h_x, err_x = self.entropy_paninski(x, method, k_max, display_flag)
            h_y, err_y = self.entropy_paninski(y, method, k_max, display_flag)
            
            if h_y <= err_y:  # If entropy is within error bounds of zero
                print(f"Warning: Cannot normalize conditional entropy: H({y_vars}) = {h_y:.4f} ± {err_y:.4f}. "
                      f"Variable has insufficient uncertainty to normalize meaningfully.")
                return 0.0, 0.0
            
            conditional_entropy = (h_xy - h_x) / h_y
            # Error propagation for division: relative error adds
            combined_error = (err_xy + err_x) / h_y + abs(conditional_entropy) * err_y / h_y
        else:
            # Only need H(X,Y) and H(X) for unnormalized
            h_xy, err_xy = self.joint_entropy_paninski(x + y, method, k_max, display_flag)
            h_x, err_x = self.entropy_paninski(x, method, k_max, display_flag)
            
            conditional_entropy = h_xy - h_x
            combined_error = err_xy + err_x
        
        return conditional_entropy, combined_error

    def mutual_information(self, vars_a, vars_b) -> float:
        """
        I(A ; B) = H(A) + H(B) − H(A, B)

        Parameters
        ----------
        vars_a, vars_b : str | Sequence[str]
        """
        if self._check_no_samples():
            return 0.0
        a = self._canon(vars_a)
        b = self._canon(vars_b)
        return self.entropy(a) + self.entropy(b) - self.joint_entropy(a + b)

    def conditional_mutual_information(self, vars_x, vars_y, given_vars_z) -> float:
        """
        I(X ; Y | Z) = H(X, Z) + H(Y, Z) − H(Z) − H(X, Y, Z)

        Parameters
        ----------
        vars_x, vars_y, given_vars_z : str | Sequence[str]
            Variable names (or sequences thereof). Can be single variables or
            tuples of variables. All combinations required by the formula must
            be tracked at construction time, in particular the triple (X, Y, Z).
        """
        if self._check_no_samples():
            return 0.0

        x = self._canon(vars_x)
        y = self._canon(vars_y)
        z = self._canon(given_vars_z)

        # Uses only joint/single entropies so we benefit from any estimator
        # plugged into those helpers. Will raise if (x+y+z) combo is untracked.
        return (
            self.joint_entropy(x + z)
            + self.joint_entropy(y + z)
            - self.entropy(z)
            - self.joint_entropy(x + y + z)
        )

    def conditional_mutual_information_paninski(
        self,
        vars_x,
        vars_y,
        given_vars_z,
        method: str = 'bub',
        k_max: int = 15,
        display_flag: bool = False,
        normalize: Union[str, None] = None,
    ) -> Tuple[float, float]:
        """
        I(X ; Y | Z) = H(X,Z) + H(Y,Z) − H(Z) − H(X,Y,Z) using Paninski estimator

        Parameters
        ----------
        vars_x, vars_y, given_vars_z : str | Sequence[str]
            Variable names (or sequences). All combinations required by the
            formula must be tracked, especially the triple (X, Y, Z).
        method : str
            'bub' for BUB estimator, 'bag' for BAG estimator
        k_max : int
            Degree of freedom parameter for BUB
        display_flag : bool
            Whether to display diagnostic information
        normalize : str or None
            If 'uncertainty-coefficient': normalize by H(Y|Z) → fraction of Y's
            remaining uncertainty (given Z) explained by X.
            If 'symmetric': normalize by sqrt(H(X|Z)·H(Y|Z)).
            If None: no normalization.

        Returns
        -------
        cmi : float
            Estimated conditional mutual information (normalized if requested)
        error_bound : float
            Approximate combined error bound (normalized if requested)
        """
        if self._check_no_samples():
            return 0.0, 0.0

        x = self._canon(vars_x)
        y = self._canon(vars_y)
        z = self._canon(given_vars_z)

        # Core CMI via Paninski entropies
        h_xz, err_xz = self.joint_entropy_paninski(x + z, method, k_max, display_flag)
        h_yz, err_yz = self.joint_entropy_paninski(y + z, method, k_max, display_flag)
        h_z, err_z = self.entropy_paninski(z, method, k_max, display_flag)
        h_xyz, err_xyz = self.joint_entropy_paninski(x + y + z, method, k_max, display_flag)

        cmi = h_xz + h_yz - h_z - h_xyz
        combined_error = err_xz + err_yz + err_z + err_xyz

        if normalize is None:
            return cmi, combined_error

        if normalize == 'uncertainty-coefficient':
            # I(X;Y|Z) / H(Y|Z)
            h_y_given_z, err_y_given_z = self.conditional_entropy_paninski(y, z, method, k_max, display_flag, normalize=False)
            if h_y_given_z <= err_y_given_z:
                print(
                    f"Warning: Cannot normalize CMI by H({vars_y}|{given_vars_z}): "
                    f"{h_y_given_z:.4f} ± {err_y_given_z:.4f}. Insufficient uncertainty."
                )
                return 0.0, 0.0
            normalized = cmi / h_y_given_z
            # Propagate error: Δ(f/g) ≈ Δf/|g| + |f|Δg/|g|^2
            norm_err = (combined_error / h_y_given_z) + (abs(cmi) * err_y_given_z) / (h_y_given_z**2)
            return normalized, norm_err

        if normalize == 'symmetric':
            # I(X;Y|Z) / sqrt(H(X|Z) H(Y|Z))
            h_x_given_z, err_x_given_z = self.conditional_entropy_paninski(x, z, method, k_max, display_flag, normalize=False)
            h_y_given_z, err_y_given_z = self.conditional_entropy_paninski(y, z, method, k_max, display_flag, normalize=False)
            if h_x_given_z <= err_x_given_z or h_y_given_z <= err_y_given_z:
                print(
                    f"Warning: Cannot symmetric-normalize CMI: H({vars_x}|{given_vars_z}) = "
                    f"{h_x_given_z:.4f} ± {err_x_given_z:.4f}, H({vars_y}|{given_vars_z}) = "
                    f"{h_y_given_z:.4f} ± {err_y_given_z:.4f}."
                )
                return 0.0, 0.0
            denom = math.sqrt(h_x_given_z * h_y_given_z)
            normalized = cmi / denom
            # Approximate error propagation for denom = sqrt(a b)
            denom_rel_err = 0.5 * (err_x_given_z / h_x_given_z + err_y_given_z / h_y_given_z)
            denom_err = denom * denom_rel_err
            norm_err = (combined_error / denom) + (abs(normalized) * denom_err / denom)
            return normalized, norm_err

        raise ValueError(f"Invalid normalization method: {normalize}")

    def mutual_information_paninski(self, vars_a, vars_b, method='bub', k_max=15, display_flag=False, normalize=None) -> Tuple[float, float]:
        """
        I(A ; B) = H(A) + H(B) − H(A, B) using Paninski estimator
        
        Parameters  
        ----------
        vars_a, vars_b : str | Sequence[str]
        method : str
            'bub' for BUB estimator, 'bag' for BAG estimator
        k_max : int
            Degree of freedom parameter for BUB
        display_flag : bool
            Whether to display diagnostic information
        normalize : str or None
            If 'uncertainty-coefficient': normalizes by H(B) to get fraction of B explained by A
            If 'symmetric': normalizes by sqrt(H(A) * H(B)) for symmetric measure
            If None: no normalization
            
        Returns
        -------
        mutual_information : float
            Estimated mutual information (normalized if normalize is specified)
        error_bound : float
            Combined error bound (approximate, normalized if normalize is specified)
        """
        if self._check_no_samples():
            return 0.0, 0.0
            
        a = self._canon(vars_a)
        b = self._canon(vars_b)
        
        # Always compute all three entropies needed (no conditional branching for efficiency)
        h_a, err_a = self.entropy_paninski(a, method, k_max, display_flag)
        h_b, err_b = self.entropy_paninski(b, method, k_max, display_flag)  
        h_ab, err_ab = self.joint_entropy_paninski(a + b, method, k_max, display_flag)
        
        mi = h_a + h_b - h_ab
        combined_error = err_a + err_b + err_ab
        
        if normalize is None:
            return mi, combined_error
        elif normalize == 'uncertainty-coefficient':
            # Fraction of B's information explained by A
            if h_b <= err_b:
                print(f"Warning: Cannot compute uncertainty coefficient: H({vars_b}) = {h_b:.4f} ± {err_b:.4f}. "
                      f"Variable B has insufficient uncertainty to normalize meaningfully.")
                return 0.0, 0.0
            normalized_mi = mi / h_b
            # Error propagation for division
            normalized_error = combined_error / h_b + abs(normalized_mi) * err_b / h_b
            return normalized_mi, normalized_error
        elif normalize == 'symmetric':
            # Symmetric normalization
            if h_a <= err_a or h_b <= err_b:
                print(f"Warning: Cannot compute symmetric normalization: H({vars_a}) = {h_a:.4f} ± {err_a:.4f}, "
                      f"H({vars_b}) = {h_b:.4f} ± {err_b:.4f}. "
                      f"Both variables must have sufficient uncertainty to normalize meaningfully.")
                return 0.0, 0.0
            denominator = math.sqrt(h_a * h_b)
            normalized_mi = mi / denominator
            # Approximate error propagation for sqrt(h_a * h_b)
            denominator_error = 0.5 * (err_a / h_a + err_b / h_b) * denominator
            normalized_error = combined_error / denominator + abs(normalized_mi) * denominator_error / denominator
            return normalized_mi, normalized_error
        else:
            raise ValueError(f"Invalid normalization method: {normalize}")

    # ------------------------------------------------------------------ #
    # Convenience
    def __repr__(self) -> str:
        tracked = ", ".join(str(c) for c in sorted(self.combos))
        return (
            f"<StreamingInfo {self.N} samples | base={self.base} "
            f"| tracked={tracked}>"
        )



    def _compute_paninski_entropy(self, histogram, method='bub', k_max=15, display_flag=False):
        """
        Compute entropy using Paninski's estimators given a histogram.
        
        Parameters:
        -----------
        histogram : array-like
            Histogram counts for each bin
        method : str
            'bub' for BUB estimator, 'bag' for BAG estimator
        k_max : int
            Degree of freedom parameter for BUB
        display_flag : bool
            Whether to display diagnostic information
            
        Returns:
        --------
        entropy : float
            Estimated entropy in bits
        error_bound : float
            Upper bound on RMS error in bits
        """
        
        histogram = np.asarray(histogram)
        N = np.sum(histogram)
        m = len(histogram)
        
        if N == 0:
            return 0.0, 0.0
        
        if m == 0:
            return 0.0, 0.0
        
        if method.lower() == 'bub':
            a, MM = self._bub_entropy_estimator(N, m, k_max, display_flag)
        elif method.lower() == 'bag':
            a, MM = self._bag_entropy_estimator(N, m, display_flag)
        else:
            raise ValueError("Method must be 'bub' or 'bag'")
        
        # Compute entropy estimate: sum over a[n_i] for each count n_i
        entropy_estimate = np.sum(a[histogram])
        
        # Convert to bits (Paninski's algorithm works in nats)
        entropy_estimate_bits = entropy_estimate / np.log(2)
        MM_bits = MM  # MM is already in bits from the algorithm
        
        return entropy_estimate_bits, MM_bits

    def _bub_entropy_estimator(self, N, m, k_max=15, display_flag=False, lambda_0=0.0):
        """
        Implements BUB entropy estimator from Paninski (2003).
        """
        
        if N < 20:
            # Use BAG function for small N
            return self._bag_entropy_estimator(N, m, display_flag, lambda_0)
        else:
            # Main BUB procedure
            if k_max > N:
                if display_flag:
                    print('Restricting k_max to be less than N...')
                k_max = N
                
            # Constants and mesh setup
            c = 80  # Constant to restrict binomial coefficients
            c = int(np.ceil(min(N, c * max(N/m, 1))))
            s = 30
            mesh = 200
            eps = (1.0/N) * 1e-10
            
            # Compute log of binomial coefficients using gammaln for numerical stability
            Ni = gammaln(N+1) - gammaln(np.arange(1, c+2)) - gammaln(N+1 - np.arange(0, c+1))
            
            # Create logarithmic mesh for p
            p = np.logspace(np.log10((1e-4)/N), np.log10(min(1, s/N) - eps), mesh)
            lp = np.log(p)
            lq = np.log(1 - p)
            
            # Compute P matrix (binomial probabilities)
            P = np.exp(Ni[:, np.newaxis] + np.arange(0, c+1)[:, np.newaxis] * lp + 
                       (N - np.arange(0, c+1))[:, np.newaxis] * lq)
            
            # Setup for variance computation
            epsm = (1.0/m) * 1e-10
            sm = s
            meshm = mesh
            step = min(1, sm/m) / meshm
            pm = np.arange(epsm, min(1, sm/m) - epsm, step)
            lpm = np.log(pm)
            lqm = np.log(1 - pm)
            
            # Compute Pm matrix
            Pm = np.exp(Ni[:, np.newaxis] + np.arange(0, c+1)[:, np.newaxis] * lpm + 
                        (N - np.arange(0, c+1))[:, np.newaxis] * lqm)
            
            # Weighting function f
            f = np.zeros_like(pm)
            mask = pm <= 1/m
            f[mask] = m
            f[~mask] = pm[~mask]**-1
            
            # Initialize coefficients
            a = np.arange(0, N+1, dtype=float) / N
            # Handle a^a where a=0: treat 0^0 = 1, so log(0^0) = log(1) = 0
            a_powered = np.ones_like(a)  # Initialize to 1 for 0^0 case
            mask = a > 0
            a_powered[mask] = a[mask] ** a[mask]
            a = -np.log(a_powered) + (1 - a) * 0.5 / N
            
            mda = np.max(np.abs(np.diff(a)))
            best_MM = np.inf
            best_a = a.copy()
            
            # Main optimization loop
            for k in range(1, min(k_max+1, N+1)):
                # Compute h_mm
                h_mm = a[k:c+1] @ P[k:, :]
                
                # Setup linear system
                XX = (m**2) * (P[:k, :] @ P[:k, :].T)
                XY = (m**2) * (P[:k, :] @ (-np.log(p**p) - h_mm))
                XY[k-1] += N * a[k-1]
                
                # Regularization matrix
                DD = 2 * np.eye(k) - np.diag(np.ones(k-1), 1) - np.diag(np.ones(k-1), -1)
                DD[0, 0] = 1
                DD[k-1, k-1] = 1
                
                # Solve regularized system
                AA = XX + N * DD
                AA[0, 0] += lambda_0
                AA[k-1, k-1] += N
                
                a[:k] = pinv(AA) @ XY
                
                # Compute bias and variance
                B = m * (a[:c+1] @ P + np.log(p**p))
                maxbias = np.max(np.abs(B))
                
                V1 = (np.arange(0, c+1) / N * (a[:c+1] - np.concatenate(([0], a[:c])))**2) @ Pm
                mmda = max(mda, np.max(np.abs(np.diff(a[:min(k+2, len(a))]))))
                MM = np.sqrt(maxbias**2 + N * min(mmda**2, 4 * np.max(f * V1))) / np.log(2)
                
                if MM < best_MM:
                    best_MM = MM
                    best_a = a.copy()
                    if display_flag:
                        print(f'Current best k = {k}; best max error = {best_MM:.3f}')
            
            return best_a, best_MM

    def _bag_entropy_estimator(self, N, m, display_flag=False, lambda_0=0.0, lambda_N=0.0):
        """
        Implements BAG entropy estimator from Paninski (2003).
        Used for small sample sizes (N < 20).
        """
        
        # Compute log factorials using gammaln
        fa = gammaln(np.arange(1, 2*N+2))
        Ni = fa[N] - fa[:N+1] - np.flip(fa[:N+1])
        
        # Create integration mesh
        p = np.arange(0, N*5+1) / (N*5)
        p = p[1:-1]  # Remove first and last elements (0 and 1)
        lp = np.log(p)
        lq = np.log(1 - p)
        
        # Build matrix of Bernoulli polynomials
        P = np.zeros((N+1, len(p)+2))
        for i in range(N+1):
            P[i, 1:-1] = Ni[i] + i * lp + (N - i) * lq
        
        P = np.exp(P)
        P[1:-1, [0, -1]] = 0
        P[0, 0] = 1
        P[-1, -1] = 1
        P[-1, 0] = 0
        P[0, -1] = 0
        
        # Add endpoints to p
        p = np.concatenate(([0], p, [1]))
        
        # Weighting function
        f = np.zeros_like(p)
        mask = p <= 1/m
        f[mask] = m
        f[~mask] = p[~mask]**-1
        
        # Setup linear system
        X = m * P.T
        XX = X.T @ X
        XY = m * (X.T @ (-np.log(p**p)))
        
        # Regularization
        DD = 2 * np.eye(N+1) - np.diag(np.ones(N), 1) - np.diag(np.ones(N), -1)
        DD[0, 0] = 1
        DD[N, N] = 1
        
        # Solve system
        AA = XX + N * DD
        AA = AA * (np.abs(AA) > np.max(np.abs(AA)) * 1e-7)
        AA[0, 0] += lambda_0
        AA[N, N] += lambda_N
        
        a = pinv(AA) @ XY
        
        # Compute error bound for finer mesh
        mesh = 10
        p_fine = np.arange(0, N*mesh+1) / (N*mesh)
        p_fine = p_fine[1:-1]
        lp_fine = np.log(p_fine)
        lq_fine = np.log(1 - p_fine)
        
        # Build P matrix for finer mesh
        P_fine = np.zeros((N+1, len(p_fine)+2))
        for i in range(N+1):
            P_fine[i, 1:-1] = Ni[i] + i * lp_fine + (N - i) * lq_fine
        
        P_fine = np.exp(P_fine)
        P_fine[1:-1, [0, -1]] = 0
        P_fine[0, 0] = 1
        P_fine[-1, -1] = 1
        P_fine[-1, 0] = 0
        P_fine[0, -1] = 0
        
        p_fine = np.concatenate(([0], p_fine, [1]))
        
        # Compute error
        Pn = a @ P_fine
        maxbias = m * np.max(np.abs(Pn + np.log(p_fine**p_fine)))
        MM = np.sqrt(maxbias**2 + N * (np.max(np.abs(np.diff(a))))**2) / np.log(2)
        
        if display_flag:
            print(f'm={m}; N={N}; max mse<{MM:.6f} bits')
        
        return a, MM


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
# Unit-style sanity checks for the StreamingInfo class.

import math
import random

def approx_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_independent_uniform():
    """
    X, Y ~ Uniform{0,1} and independent
      • H(X) = H(Y) = 1 bit
      • H(X,Y) = 2 bits
      • I(X;Y) = 0
      • H(Y|X) = 1
    """
    N = 10_000
    smi = StreamingInfo(["X", "Y"], base=2)

    for _ in range(N):
        x = random.randint(0, 1)
        y = random.randint(0, 1)
        smi.update({"X": x, "Y": y})

    HX = smi.entropy("X")
    HY = smi.entropy("Y")
    HXY = smi.joint_entropy(("X", "Y"))
    IXY = smi.mutual_information("X", "Y")
    HY_given_X = smi.conditional_entropy("Y", "X")

    assert approx_equal(HX, 1.0, 1e-2)
    assert approx_equal(HY, 1.0, 1e-2)
    assert approx_equal(HXY, 2.0, 2e-2)
    assert approx_equal(IXY, 0.0, 1e-2)
    assert approx_equal(HY_given_X, 1.0, 1e-2)
    print("✓ independent_uniform passed")


def test_copy_variable():
    """
    Y = X  (perfect copy)
      • X, Y ∈ {0,1} equally likely
      • H(X) = H(Y) = 1
      • H(X,Y) = 1          (because outcomes are (0,0) or (1,1))
      • I(X;Y) = 1
      • H(Y|X) = 0
    """
    N = 10_000
    smi = StreamingInfo(["X", "Y"], base=2)

    for _ in range(N):
        x = random.randint(0, 1)
        smi.update({"X": x, "Y": x})

    HX = smi.entropy("X")
    HY = smi.entropy("Y")
    HXY = smi.joint_entropy(("X", "Y"))
    IXY = smi.mutual_information("X", "Y")
    HY_given_X = smi.conditional_entropy("Y", "X")

    assert approx_equal(HX, 1.0, 1e-2)
    assert approx_equal(HY, 1.0, 1e-2)
    assert approx_equal(HXY, 1.0, 1e-2)
    assert approx_equal(IXY, 1.0, 1e-2)
    assert approx_equal(HY_given_X, 0.0, 1e-2)
    print("✓ copy_variable passed")


def test_three_variables():
    """
    Example with three tracked variables to show combos work.

    Let
        Z = X XOR Y  (mod-2 sum)
    with X, Y independent Bernoulli(0.5).

    Facts:
      • H(X)=H(Y)=H(Z)=1
      • H(X,Y)=2   (independent)
      • H(X,Z)=2   (also independent)
      • I(X;Z)=0
    """
    N = 20_000
    smi = StreamingInfo(
        ["X", "Y", "Z"],
        combos_to_track=[("X",), ("Y",), ("Z",), ("X", "Z")],
        base=2,
    )

    for _ in range(N):
        x = random.randint(0, 1)
        y = random.randint(0, 1)
        z = x ^ y
        smi.update({"X": x, "Y": y, "Z": z})

    assert approx_equal(smi.entropy("X"), 1.0, 1e-2)
    assert approx_equal(smi.entropy("Z"), 1.0, 1e-2)
    assert approx_equal(smi.joint_entropy(("X", "Z")), 2.0, 2e-2)
    assert approx_equal(smi.mutual_information("X", "Z"), 0.0, 1e-2)
    print("✓ three_variables passed")


def test_no_samples():
    """
    Test that querying metrics with no samples returns 0 and prints a warning.
    """
    smi = StreamingInfo(["X", "Y"], base=2)
    
    # Test that all metrics return 0 when no samples added
    print("Testing no samples case (expect warnings)...")
    assert smi.entropy("X") == 0.0
    assert smi.joint_entropy(("X", "Y")) == 0.0
    assert smi.conditional_entropy("Y", "X") == 0.0
    assert smi.mutual_information("X", "Y") == 0.0
    print("✓ no_samples passed")


if __name__ == "__main__":
    random.seed(0)
    test_independent_uniform()
    test_copy_variable()
    test_three_variables()
    test_no_samples()
    print("All tests ✅")
