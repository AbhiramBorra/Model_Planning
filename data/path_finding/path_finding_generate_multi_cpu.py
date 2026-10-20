#!/usr/bin/env python3
"""
Synthetic path‑finding graph generator + optional visualiser.

Run:
    python generate.py --config cfg.json

The cfg.json (or .yml) schema is documented in the README.  All keys are
optional – sensible defaults apply.

Example minimal config
----------------------
{
  "output_dir": "data/path_samples",
  "dataset_size": 100,
  "seed": 7,
  "max_nodes": 250, # [1, 250] 
  "n_paths": 2,          
  "n_decoy": 2,          
  "min_path_len": 3,
  "max_path_len": 6,    
  "seed": 42,
  "branch_lambda": [1.5, 1.0, 0.6, 0.3],
  "max_children": 4
}

# max_children:
# This is the number of additional children per node. 
# I mean on a decoy path each node can have at most two additional children (1 already exitst before adding these because of the path)
# Thus each node on a decoy or correct path can have at most max_children + 1 children
#
# as for the start node, it can have at most max_children - n_correct - n_decoy additional children, so it can have at most max_children children.


python data/path_finding/path_finding_generate.py --config data/path_finding/PF_10_config.json
"""

import argparse, json, random, pathlib, textwrap, sys
from typing import Dict, List
import math
import multiprocessing as mp
import os
import threading
from queue import Queue

# ---------------------------------------------------------------------------
# optional deps (only needed for draw_images = True)
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    nx = None
    plt = None

try:
    import yaml   # graceful fall‑back to JSON if unavailable
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# optional numpy for Poisson branching (optional)
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------------
# Import visualization functions (only needed when visualize=True)
# ---------------------------------------------------------------------------
def import_visualization_functions():
    """Import visualization functions when needed to avoid circular imports."""
    try:
        from visualize_saved_graphs import load_dataset, reconstruct_graph_from_record, enhanced_visualise_graph
        return load_dataset, reconstruct_graph_from_record, enhanced_visualise_graph
    except ImportError as e:
        print(f"⚠️  Could not import visualization functions: {e}", file=sys.stderr)
        return None, None, None


# ---------------------------------------------------------------------------
# Block size analysis integration
# ---------------------------------------------------------------------------
def run_block_size_analysis(out_dir: pathlib.Path, dataset_name: str):
    """
    Automatically run block size analysis on the generated validation dataset.
    
    Args:
        out_dir: Output directory containing the dataset
        dataset_name: Name of the dataset (e.g., 'val', 'test')
    """
    print(f"🔍 Running automatic block size analysis for {dataset_name} dataset...")
    
    try:
        # Import the analysis script as a module
        import subprocess
        import os
        
        # Get the path to the analysis script
        # Assume it's in the same directory as the training scripts
        analysis_script = pathlib.Path(__file__).parent.parent.parent / "path_finding_analyze_block_size.py"
        
        if not analysis_script.exists():
            print(f"⚠️  Block size analysis script not found at: {analysis_script}")
            return
        
        # Run the analysis script
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(analysis_script),
            "--dataset_dir", str(out_dir),
            "--dataset_name", dataset_name,
            "--max_samples", "2000",
            "--output_name", f"block_size_analysis_{dataset_name}.txt"
        ]
        
        print(f"📊 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir.parent.parent))
        
        if result.returncode == 0:
            print(f"✅ Block size analysis completed successfully!")
            print(f"📋 Results saved to: {out_dir}/block_size_analysis_{dataset_name}.txt")
            
            # Print summary from stdout if available
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                # Look for the analysis summary
                summary_start = -1
                for i, line in enumerate(lines):
                    if "ANALYSIS SUMMARY" in line:
                        summary_start = i
                        break
                
                if summary_start >= 0:
                    print("\n📊 Block Size Analysis Summary:")
                    print("-" * 50)
                    for line in lines[summary_start:]:
                        if line.strip() and not line.startswith("="):
                            print(f"  {line}")
                        if "Analysis complete!" in line:
                            break
        else:
            print(f"⚠️  Block size analysis failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
                
    except Exception as e:
        print(f"⚠️  Error running block size analysis: {e}")
        import traceback
        traceback.print_exc()


# ===========================================================================
#  Small undirected simple graph helper
# ===========================================================================
class Graph:
    """
    Tiny wrapper around an undirected graph using adjacency sets.
    Nodes are ints [1, n_nodes].
    """
    def __init__(self):
        """
        Adjacency dict grows dynamically.
        Node 1 is the start, node 2 is the real goal.
        Every new node gets the next positive integer ID.
        """
        self.adj: Dict[int, set[int]] = {1: set(), 2: set()}
        self._next_id = 3

    # ————————————————————————————————————————————
    def new_node(self) -> int:
        """Create and return a fresh node ID."""
        nid = self._next_id
        self._next_id += 1
        self.adj[nid] = set()
        return nid

    # ————————————————————————————————————————————
    def add_edge(self, u: int, v: int):
        if u == v:
            return
        self.adj[u].add(v)
        self.adj[v].add(u)

    def edges(self):
        seen = set()
        for u, nbrs in self.adj.items():
            for v in nbrs:
                if (v, u) not in seen:
                    seen.add((u, v))
        return list(seen)

    # ————————————————————————————————————————————
    def to_networkx(self):
        """Return a networkx.Graph() copy (for drawing)."""
        if nx is None:
            raise RuntimeError("networkx missing ‑ cannot visualise.")
        g = nx.Graph()
        g.add_nodes_from(self.adj.keys())
        g.add_edges_from(self.edges())
        return g


# ===========================================================================
#  Generation logic
# ===========================================================================
def sample_path(pool: List[int], length: int, rng: random.Random) -> List[int]:
    """Pop `length` unique nodes from pool and return them as a list."""
    path = rng.sample(pool, length)
    for n in path:
        pool.remove(n)
    return path

def generate_single_graph_record(cfg: Dict, graph_idx: int) -> Dict:
    """Generate a single graph and return the final record (not just the bundle)."""
    while True:
        bundle = build_graph(cfg, graph_idx)
        if bundle is not None:
            break   # size OK
    
    # Get random number generator with reproducible seed
    rng = random.Random(cfg.get("seed"))
    if cfg.get("seed") is not None:
        # Use a different seed for relabeling to maintain consistency
        rng.seed(cfg["seed"] + graph_idx * 1000000)  # Large offset to avoid overlap
    
    # --- random relabeling (break arithmetic pattern) ----------------
    internal_nodes = [n for n in bundle["graph"].adj.keys() if n not in {1, 2}]
    dict_cap = cfg.get("max_nodes", 1000)
    if len(internal_nodes) > dict_cap - 2:
        raise ValueError("Graph has more nodes than dictionary allows.")
    new_labels_pool = rng.sample(range(3, dict_cap + 1), len(internal_nodes))
    mapping = {1: 1, 2: 2, **{old: new for old, new in zip(internal_nodes, new_labels_pool)}}

    # remap edges
    edges_mapped = [(mapping[u], mapping[v]) for u, v in bundle["graph"].edges()]
    rng.shuffle(edges_mapped)  # Use same rng for consistency

    # remap paths
    def remap_path(path):
        return [str(mapping[n]) for n in path]

    correct_mapped = [remap_path(p) for p in bundle["paths"]]
    decoy_mapped   = [remap_path(p) for p in bundle["decoys"]]

    record = {
        "graph_id": graph_idx,
        "start": "1",
        "goal": "2",
        "edges": [f"{u}-{v}" for u, v in edges_mapped],
        "correct_paths": correct_mapped,
        "decoy_paths": decoy_mapped
    }
    return record


def build_graph(cfg: Dict, graph_idx: int) -> Dict:
    """
    Create ONE graph + paths.

    Returns a dict:
      { 'graph': Graph, 'start': s, 'goal': g, 'paths': [[s,…,g], …] }
    """
    rng = random.Random(cfg.get("seed"))              # reproducible
    if cfg.get("seed") is not None:
        rng.seed(cfg["seed"] + graph_idx)

    # Number of correct and decoy trunks
    n_correct  = cfg.get("n_paths", 2)
    n_decoy    = cfg.get("n_decoy", 2)
    min_len = cfg.get("min_path_len", cfg.get("path_len", 4))
    max_len = cfg.get("max_path_len", min_len)
    if max_len < min_len:
        min_len, max_len = max_len, min_len  # swap to keep sane
    # Branching controls (probabilistic, per-layer Poisson)
    branch_lambdas = cfg.get("branch_lambda", [1.2, 0.8, 0.5])
    max_children   = cfg.get("max_children", 4)   # hard cap per node
    max_branch_depth = len(branch_lambdas)

    max_nodes_cap = cfg.get("max_nodes", 1000)     # global dictionary size

    # 1) fixed start/goal
    s, g = 1, 2            # always!

    trunks = []
    correct_paths = []
    decoy_paths = []
    decoy_end_nodes = []
    correct_lengths = []

    # Graph grows dynamically; start with nodes 1 & 2 already present
    g_obj = Graph()

    # Create correct trunks
    for _ in range(n_correct):
        plen = rng.randint(min_len, max_len)
        correct_lengths.append(plen)
        mids = [g_obj.new_node() for _ in range(plen)]
        path = [s] + mids + [g]
        trunks.append(path)
        correct_paths.append(path)
    # Create decoy trunks — decoy_i length = correct_lengths[i] if exists, else random pick
    for i in range(n_decoy):
        if correct_lengths:         # should always be true
            if i < len(correct_lengths):
                dec_len = correct_lengths[i]          # mirror the i‑th correct path
            else:
                dec_len = rng.choice(correct_lengths) # fall back to random among l_i
        else:
            dec_len = rng.randint(min_len, max_len)   # safety fallback
        mids = [g_obj.new_node() for _ in range(dec_len)]
        dec_end = g_obj.new_node()
        decoy_end_nodes.append(dec_end)

        path = [s] + mids + [dec_end]
        trunks.append(path)
        decoy_paths.append(path)
    # Build graph edges for all trunks
    for path in trunks:
        for u, v in zip(path, path[1:]):
            g_obj.add_edge(u, v)
    # ----------------------------------------------------------------
    #  Layer‑wise probabilistic branching (Poisson)
    # ----------------------------------------------------------------
    # Collect unique nodes from all trunks (avoid duplicates like node 1 appearing 4 times)
    unique_trunk_nodes = set()
    for path in trunks:
        for node in path:
            if node not in {g} | set(decoy_end_nodes):  # exclude terminals
                unique_trunk_nodes.add(node)
    queue = [(node, 0) for node in unique_trunk_nodes]

    # Track how many branching children each node has (separate from trunk connections)
    branching_children_count = {}

    def poisson_int(lmbd):
        """Draw from Poisson(lambda) without numpy fallback."""
        if np is not None:
            return int(np.random.poisson(lmbd))
        # simple Knuth algorithm
        L = math.exp(-lmbd)
        k, p_ = 0, 1.0
        while p_ > L:
            k += 1
            p_ *= rng.random()
        return k - 1

    while queue:
        parent, depth = queue.pop(0)
        if depth >= max_branch_depth:
            continue
        
        # Special handling for start node (node 1): account for trunk connections
        if parent == s:  # s is the start node (1)
            trunk_children = n_correct + n_decoy
            max_branching_for_start = max(max_children - trunk_children, 0)
            current_branching_children = branching_children_count.get(parent, 0)
            if current_branching_children >= max_branching_for_start:
                continue
            max_new_children = max_branching_for_start - current_branching_children
        else:
            # For other nodes, use the original logic (only count branching children)
            current_branching_children = branching_children_count.get(parent, 0)
            if current_branching_children >= max_children:
                continue  # Skip if already at or above max_children
            max_new_children = max_children - current_branching_children
            
        lam = branch_lambdas[depth]
        n_children = min(poisson_int(lam), max_new_children)
        
        for _ in range(n_children):
            child = g_obj.new_node()
            g_obj.add_edge(parent, child)
            queue.append((child, depth + 1))
            
        # Update the branching children count
        branching_children_count[parent] = current_branching_children + n_children

    # drop this graph if it exceeds the cap
    if len(g_obj.adj) >= max_nodes_cap:
        return None

    return {"graph": g_obj, "start": s, "goal": g, "paths": correct_paths, "decoys": decoy_paths}


# ===========================================================================
#  Config helpers
# ===========================================================================
def load_cfg(path: pathlib.Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
            return yaml.safe_load(f)
        return json.load(f)


# -----------------------------------------------------------------------
#  Dataset writer (iterative JSONL file)
# -----------------------------------------------------------------------
def write_dataset(records, out_dir: pathlib.Path, dataset_name: str):
    """Write records to JSONL file (backward compatibility)."""
    out_path = out_dir / f"{dataset_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")
    print(f"💾  Saved {len(records)} samples → {out_path}")


def append_records_to_dataset(records, out_dir: pathlib.Path, dataset_name: str):
    """Append records to JSONL file for iterative saving."""
    out_path = out_dir / f"{dataset_name}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")


def init_dataset_file(out_dir: pathlib.Path, dataset_name: str):
    """Initialize/clear the dataset file."""
    out_path = out_dir / f"{dataset_name}.jsonl"
    # Create empty file or truncate existing file
    with out_path.open("w", encoding="utf-8") as f:
        pass  # Just create/truncate the file


# ===========================================================================
#  Multiprocessing worker functions
# ===========================================================================
def worker_process(cfg: Dict, start_idx: int, end_idx: int, worker_id: int, temp_dir: pathlib.Path):
    """Worker process that generates graphs and saves them to a temporary file."""
    temp_file = temp_dir / f"worker_{worker_id}.jsonl"
    
    with temp_file.open("w", encoding="utf-8") as f:
        for idx in range(start_idx, end_idx):
            try:
                record = generate_single_graph_record(cfg, idx)
                json.dump(record, f)
                f.write("\n")
                
                # Progress reporting (every 100 graphs)
                if (idx - start_idx + 1) % 100 == 0:
                    print(f"⚡ Worker {worker_id}: Generated {idx - start_idx + 1}/{end_idx - start_idx} graphs")
                    
            except Exception as e:
                print(f"⚠️  Worker {worker_id} failed on graph {idx}: {e}")
                raise
    
    print(f"✅ Worker {worker_id}: Completed {end_idx - start_idx} graphs → {temp_file}")
    return str(temp_file)


def merge_worker_files(temp_files: List[str], out_dir: pathlib.Path, dataset_name: str, total_samples: int):
    """Merge temporary worker files into final dataset file."""
    out_path = out_dir / f"{dataset_name}.jsonl"
    
    print(f"🔄 Merging {len(temp_files)} worker files into {out_path}...")
    
    try:
        with out_path.open("w", encoding="utf-8") as out_f:
            records_written = 0
            for i, temp_file in enumerate(temp_files):
                temp_path = pathlib.Path(temp_file)
                print(f"📂 Processing worker file {i+1}/{len(temp_files)}: {temp_path}")
                
                if temp_path.exists():
                    try:
                        with temp_path.open("r", encoding="utf-8") as in_f:
                            file_records = 0
                            for line in in_f:
                                if line.strip():  # Skip empty lines
                                    out_f.write(line)
                                    records_written += 1
                                    file_records += 1
                        print(f"✅ Merged {file_records} records from worker {i+1}")
                        
                        # Clean up temporary file
                        temp_path.unlink()
                        print(f"🗑️  Cleaned up {temp_path}")
                        
                    except Exception as e:
                        print(f"❌ Error reading worker file {temp_path}: {e}")
                        raise
                else:
                    print(f"⚠️  Worker file {temp_path} does not exist!")
        
        print(f"💾 Successfully merged {records_written} samples → {out_path}")
        
        if records_written != total_samples:
            print(f"⚠️  Warning: Expected {total_samples} samples, but merged {records_written}")
        
        return records_written
        
    except Exception as e:
        print(f"❌ Error during file merging: {e}")
        raise


def get_optimal_worker_count():
    """Determine optimal number of worker processes."""
    cpu_count = mp.cpu_count()
    # Use all CPUs but leave one for the main process
    optimal_workers = max(1, cpu_count - 1)
    print(f"💻 Detected {cpu_count} CPUs, using {optimal_workers} worker processes")
    return optimal_workers


# ===========================================================================
#  CLI entry
# ===========================================================================
def main():   
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Synthetic path‑finding graph generator.
            ------------------------------------------------------------
            Provide a JSON or YAML config via --config.  Any missing
            keys take default values (see README).
            """))
    ap.add_argument("--config", required=True, help="Config file (JSON/YAML)")
    ap.add_argument("--visualize", action='store_true', help="Enable visualization (overrides config setting)")
    ap.add_argument("--no-visualize", action='store_true', help="Disable visualization (overrides config setting)")
    ap.add_argument("--skip-block-analysis", action='store_true', help="Skip automatic block size analysis for validation datasets")
    ap.add_argument("--num-workers", type=int, help="Number of worker processes (default: auto-detect)")
    ap.add_argument("--single-process", action='store_true', help="Force single-process mode (useful for debugging)")
    args = ap.parse_args()
    
    cfg = load_cfg(pathlib.Path(args.config))
    #cfg = load_cfg(pathlib.Path("path_config.json"))
    out_dir = pathlib.Path(cfg.get("output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = cfg.get("dataset_name", "dataset")

    # Save config to output directory
    config_path = out_dir / f"config_{dataset_name}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"💾  Saved config → {config_path}")

    ds_size  = cfg.get("dataset_size", 1000)
    # Handle visualization flags with priority: --no-visualize > --visualize > config
    if args.no_visualize:
        visualize = False
        print("🚫 Visualization disabled via --no-visualize flag")
    elif args.visualize:
        visualize = True
        print("🎨 Visualization enabled via --visualize flag")
    else:
        visualize = bool(cfg.get("visualize", False))
    seed      = cfg.get("seed")
    
    # Determine worker count
    if args.single_process:
        num_workers = 1
        print("🔧 Single-process mode enabled")
    elif args.num_workers:
        num_workers = args.num_workers
        print(f"🔧 Using {num_workers} worker processes (user specified)")
    else:
        num_workers = get_optimal_worker_count()

    # show quick summary
    print(f"⏳  Generating {ds_size} samples into {out_dir}/{dataset_name}.jsonl (seed={seed}).  Visualise={visualize}")
    print(f"⚡ Using {num_workers} worker process(es)")

    if num_workers == 1:
        # Single-process mode (original behavior with iterative saving)
        print("📝 Single-process mode: generating graphs sequentially...")
        init_dataset_file(out_dir, dataset_name)
        
        batch_size = 100  # Save every 100 graphs to reduce memory usage
        batch_records = []
        
        for idx in range(ds_size):
            record = generate_single_graph_record(cfg, idx)
            batch_records.append(record)
            
            # Save batch when it reaches batch_size or at the end
            if len(batch_records) >= batch_size or idx == ds_size - 1:
                append_records_to_dataset(batch_records, out_dir, dataset_name)
                print(f"💾 Saved batch: {idx + 1 - len(batch_records) + 1}-{idx + 1}/{ds_size}")
                batch_records.clear()  # Free memory
                
        total_records = ds_size
    else:
        # Multi-process mode
        print(f"🚀 Multi-process mode: spawning {num_workers} worker processes...")
        
        # Create temporary directory for worker files
        temp_dir = out_dir / "temp_workers"
        temp_dir.mkdir(exist_ok=True)
        
        # Calculate work distribution
        graphs_per_worker = ds_size // num_workers
        remainder = ds_size % num_workers
        
        # Create work chunks
        work_chunks = []
        start_idx = 0
        for worker_id in range(num_workers):
            # Give remainder graphs to first few workers
            chunk_size = graphs_per_worker + (1 if worker_id < remainder else 0)
            end_idx = start_idx + chunk_size
            work_chunks.append((start_idx, end_idx, worker_id))
            start_idx = end_idx
            
        print(f"📊 Work distribution: {[end - start for start, end, _ in work_chunks]} graphs per worker")
        
        # Start worker processes
        with mp.Pool(num_workers) as pool:
            # Submit all worker tasks
            worker_results = []
            for start_idx, end_idx, worker_id in work_chunks:
                result = pool.apply_async(
                    worker_process, 
                    (cfg, start_idx, end_idx, worker_id, temp_dir)
                )
                worker_results.append(result)
            
            # Wait for all workers to complete and collect temp file paths
            temp_files = []
            print(f"⏳ Waiting for {len(worker_results)} workers to complete...")
            for i, result in enumerate(worker_results):
                try:
                    print(f"📥 Collecting result from worker {i}...")
                    temp_file = result.get(timeout=3600)  # 1 hour timeout per worker
                    temp_files.append(temp_file)
                    print(f"✅ Worker {i} completed successfully: {temp_file}")
                except mp.TimeoutError:
                    print(f"⏰ Worker {i} timed out after 1 hour")
                    raise
                except Exception as e:
                    print(f"❌ Worker {i} failed: {e}")
                    raise
            
            print(f"🎯 All {len(worker_results)} workers completed. Proceeding to merge files...")
        
        # Merge all worker files into final dataset
        total_records = merge_worker_files(temp_files, out_dir, dataset_name, ds_size)
        
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except OSError:
            print(f"⚠️  Could not remove temp directory {temp_dir} (may contain leftover files)")

    print(f"✅  Dataset generation completed! Generated {total_records} samples.")

    # Automatically run block size analysis for validation datasets
    if dataset_name.lower() in ['val', 'validation', 'test'] and not args.skip_block_analysis:
        run_block_size_analysis(out_dir, dataset_name)

    # Add visualization if requested
    if visualize:
        print(f"🎨  Generating visualizations for first 10 samples...")
        load_dataset, reconstruct_graph_from_record, enhanced_visualise_graph = import_visualization_functions()
        
        if all(func is not None for func in [load_dataset, reconstruct_graph_from_record, enhanced_visualise_graph]):
            try:
                # Create visualization directory
                viz_dir = out_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                # Load only the first 10 records efficiently (don't load entire dataset)
                dataset_file = out_dir / f"{dataset_name}.jsonl"
                sample_records = []
                
                print(f"📖 Reading first 10 records from {dataset_file}...")
                with dataset_file.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # Only read first 10 lines
                            break
                        if line.strip():
                            sample_records.append(json.loads(line))
                
                num_to_visualize = len(sample_records)
                print(f"🖼️  Creating {num_to_visualize} visualization(s)...")
                
                for i, record in enumerate(sample_records):
                    bundle = reconstruct_graph_from_record(record)
                    
                    # Create descriptive name based on dataset type
                    viz_dataset_name = f"{dataset_name}_training" if "train" in dataset_name.lower() else f"{dataset_name}_samples"
                    enhanced_visualise_graph(bundle, viz_dir, i, viz_dataset_name)
                    print(f"🖼️  Generated visualization {i+1}/{num_to_visualize}")
                
                print(f"✅  Visualizations saved to {viz_dir}")
                print(f"🏷️  Image pattern: {viz_dataset_name}_sample_{{index:06d}}.png")
                
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  Visualization functions not available - skipping visualization", file=sys.stderr)

    print("✅  Done.")

if __name__ == "__main__":
    main()