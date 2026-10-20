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
#  Dataset writer (single JSONL file)
# -----------------------------------------------------------------------
def write_dataset(records, out_dir: pathlib.Path, dataset_name: str):
    out_path = out_dir / f"{dataset_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")
    print(f"💾  Saved {len(records)} samples → {out_path}")


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
    ap.add_argument("--skip-block-analysis", action='store_true', help="Skip automatic block size analysis for validation datasets")
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
    visualize = args.visualize or bool(cfg.get("visualize", False))  # Command line overrides config
    seed      = cfg.get("seed")

    # show quick summary
    print(f"⏳  Generating {ds_size} samples into {out_dir}/{dataset_name}.jsonl (seed={seed}).  Visualise={visualize}")

    rng = random.Random(seed)
    records = []

    for idx in range(ds_size):
        while True:
            bundle = build_graph(cfg, idx)
            if bundle is not None:
                break   # size OK
        # --- random relabeling (break arithmetic pattern) ----------------
        internal_nodes = [n for n in bundle["graph"].adj.keys() if n not in {1, 2}]
        dict_cap = cfg.get("max_nodes", 1000)
        if len(internal_nodes) > dict_cap - 2:
            raise ValueError("Graph has more nodes than dictionary allows.")
        new_labels_pool = rng.sample(range(3, dict_cap + 1), len(internal_nodes))
        mapping = {1: 1, 2: 2, **{old: new for old, new in zip(internal_nodes, new_labels_pool)}}

        # remap edges
        edges_mapped = [(mapping[u], mapping[v]) for u, v in bundle["graph"].edges()]
        random.shuffle(edges_mapped)

        # remap paths
        def remap_path(path):
            return [str(mapping[n]) for n in path]

        correct_mapped = [remap_path(p) for p in bundle["paths"]]
        decoy_mapped   = [remap_path(p) for p in bundle["decoys"]]

        record = {
            "graph_id": idx,
            "start": "1",
            "goal": "2",
            "edges": [f"{u}-{v}" for u, v in edges_mapped],
            "correct_paths": correct_mapped,
            "decoy_paths": decoy_mapped
        }
        records.append(record)

    write_dataset(records, out_dir, dataset_name=dataset_name)

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
                
                # Load the first 10 records for visualization
                dataset_file = f"{dataset_name}.jsonl"
                sample_records = load_dataset(out_dir, dataset_file)
                num_to_visualize = min(10, len(sample_records))
                
                print(f"🖼️  Creating {num_to_visualize} visualization(s)...")
                
                for i in range(num_to_visualize):
                    record = sample_records[i]
                    bundle = reconstruct_graph_from_record(record)
                    
                    # Create descriptive name based on dataset type
                    viz_dataset_name = f"{dataset_name}_training" if "train" in dataset_name.lower() else f"{dataset_name}_samples"
                    enhanced_visualise_graph(bundle, viz_dir, i, viz_dataset_name)
                
                print(f"✅  Visualizations saved to {viz_dir}")
                print(f"🏷️  Image pattern: {viz_dataset_name}_sample_{{index:06d}}.png")
                
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}", file=sys.stderr)
        else:
            print("⚠️  Visualization functions not available - skipping visualization", file=sys.stderr)

    print("✅  Done.")

if __name__ == "__main__":
    main()