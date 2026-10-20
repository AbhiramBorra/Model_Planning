#!/usr/bin/env python3
"""
Visualize saved graphs from the dataset.jsonl file.

Usage:
    # Visualize all graphs (default behavior)
    python visualize_saved_graphs.py --data_dir data/path_samples
    
    # Visualize graphs in a range
    python visualize_saved_graphs.py --data_dir data/path_samples --start_idx 0 --end_idx 5
    
    # Visualize specific graphs by indices
    python visualize_saved_graphs.py --data_dir data/path_samples --indices 1 5 10 23 50

    # Specify custom dataset file
    python visualize_saved_graphs.py --data_dir data/path_samples --file_name train.jsonl
    
    # Visualize train dataset with specific indices
    python visualize_saved_graphs.py --data_dir data/path_samples --file_name train.jsonl --indices 0 1 2

    # Visualize test dataset in a range
    python visualize_saved_graphs.py --data_dir data/path_samples --file_name test.jsonl --start_idx 0 --end_idx 10

python data/path_finding/visualize_saved_graphs.py --data_dir PF_10/samples --file_name val.jsonl --start_idx 0 --end_idx 5
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List

# Import the Graph class from the main module
from path_finding_generate import Graph

# ---------------------------------------------------------------------------
# optional deps (only needed for draw_images = True)
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    nx = None
    plt = None
    mcolors = None


def load_dataset(data_dir: pathlib.Path, file_name: str) -> List[Dict]:
    """Load the dataset from the JSONL file."""
    dataset_path = data_dir / file_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    records = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    return records


def reconstruct_graph_from_record(record: Dict) -> Dict:
    """Reconstruct a graph bundle from a dataset record."""
    # Parse edges
    edges = []
    max_node = 0
    for edge_str in record["edges"]:
        u, v = map(int, edge_str.split("-"))
        edges.append((u, v))
        max_node = max(max_node, u, v)
    
    # Create graph - Graph() constructor no longer takes parameters
    graph = Graph()
    
    # Ensure all nodes exist in the graph before adding edges
    # The Graph class starts with nodes 1 and 2, so we need to add any missing nodes
    for edge_u, edge_v in edges:
        # Add nodes to adjacency dict if they don't exist
        if edge_u not in graph.adj:
            graph.adj[edge_u] = set()
        if edge_v not in graph.adj:
            graph.adj[edge_v] = set()
    
    # Now add all edges
    for u, v in edges:
        graph.add_edge(u, v)
    
    # Convert string paths to integer paths for proper matching
    correct_paths_int = []
    for path in record["correct_paths"]:
        correct_paths_int.append([int(node) for node in path])
    
    decoy_paths_int = []
    if "decoy_paths" in record:
        for path in record["decoy_paths"]:
            decoy_paths_int.append([int(node) for node in path])
    
    # Create bundle - include decoy paths if they exist
    bundle = {
        "graph": graph,
        "start": int(record["start"]),  # Convert to integer
        "goal": int(record["goal"]),    # Convert to integer
        "paths": correct_paths_int      # Use integer paths
    }
    
    # Add decoy paths if they exist in the record
    if decoy_paths_int:
        bundle["decoys"] = decoy_paths_int
    
    return bundle


def enhanced_visualise_graph(bundle: Dict, out_dir: pathlib.Path, idx: int, dataset_name: str = "dataset"):
    """Save a PNG visualisation of the graph with highlighted solution paths and decoy paths."""
    if nx is None or plt is None:
        print("⚠️  networkx / matplotlib not available; skipping visualisation.", file=sys.stderr)
        return

    g_nx = bundle["graph"].to_networkx()
    s, g = bundle["start"], bundle["goal"]
    solution_paths = bundle["paths"]
    decoy_paths = bundle.get("decoys", [])

    # Node colours: start (green), goal (red), decoy endpoints (orange), others (lightblue)
    decoy_endpoints = set()
    if decoy_paths:
        for decoy in decoy_paths:
            if len(decoy) > 0:
                decoy_endpoints.add(decoy[-1])  # Last node in decoy path (now integer)
    
    node_colors = []
    for node in g_nx.nodes():
        if node == s:
            node_colors.append("green")
        elif node == g:
            node_colors.append("red")
        elif node in decoy_endpoints:
            node_colors.append("orange")
        else:
            node_colors.append("#1f78b4")

    # Prepare edge coloring
    # Define colors for solution paths (bright, distinct colors)
    path_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    # Define colors for decoy paths (darker, muted colors)
    decoy_colors = ['#8B0000', '#006400', '#191970', '#556B2F', '#8B4513', '#483D8B', '#2F4F4F']
    
    # Create sets of edges for each solution path
    # Now that paths contain integers, edge matching will work correctly
    path_edge_sets = []
    for i, path in enumerate(solution_paths):
        path_edges = set()
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]  # Now integers, not strings
            # Add both orientations since networkx treats edges as undirected
            path_edges.add((min(u, v), max(u, v)))
        path_edge_sets.append(path_edges)
    
    # Create sets of edges for each decoy path
    decoy_edge_sets = []
    for i, path in enumerate(decoy_paths):
        path_edges = set()
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]  # Now integers, not strings
            # Add both orientations since networkx treats edges as undirected
            path_edges.add((min(u, v), max(u, v)))
        decoy_edge_sets.append(path_edges)
    
    # Get all edges in the graph
    all_edges = list(g_nx.edges())
    
    # Categorize edges into solution, decoy, and regular edges
    solution_edges = []
    decoy_edges = []
    regular_edges = []
    
    for edge in all_edges:
        u, v = edge
        normalized_edge = (min(u, v), max(u, v))  # integers from networkx
        
        # Check if this edge belongs to any solution path
        found_in_solution = -1
        for i, path_edge_set in enumerate(path_edge_sets):
            if normalized_edge in path_edge_set:  # Now integer-to-integer comparison
                found_in_solution = i
                break
        
        # Check if this edge belongs to any decoy path
        found_in_decoy = -1
        for i, decoy_edge_set in enumerate(decoy_edge_sets):
            if normalized_edge in decoy_edge_set:  # Now integer-to-integer comparison
                found_in_decoy = i
                break
        
        if found_in_solution >= 0:
            # This edge is part of a solution path
            solution_edges.append((edge, found_in_solution))
        elif found_in_decoy >= 0:
            # This edge is part of a decoy path
            decoy_edges.append((edge, found_in_decoy))
        else:
            # Regular edge
            regular_edges.append(edge)

    pos = nx.spring_layout(g_nx, seed=idx)  # deterministic per graph id
    plt.figure(figsize=(12, 8))
    
    # Draw regular edges first (so they appear behind solution/decoy edges)
    if regular_edges:
        nx.draw_networkx_edges(g_nx, pos, edgelist=regular_edges, 
                              edge_color="#CCCCCC", width=1.0, alpha=0.5)
    
    # Draw solution path edges (solid lines)
    for edge, path_idx in solution_edges:
        color = path_colors[path_idx % len(path_colors)]
        nx.draw_networkx_edges(g_nx, pos, edgelist=[edge], 
                              edge_color=color, width=3.0, alpha=0.9)
    
    # Draw decoy path edges (dashed lines)
    for edge, decoy_idx in decoy_edges:
        color = decoy_colors[decoy_idx % len(decoy_colors)]
        nx.draw_networkx_edges(g_nx, pos, edgelist=[edge], 
                              edge_color=color, width=2.5, alpha=0.8, style='dashed')
    
    # Draw nodes
    nx.draw_networkx_nodes(g_nx, pos, node_color=node_colors, node_size=500, linewidths=1.0, edgecolors='black')
    
    # Draw labels
    nx.draw_networkx_labels(g_nx, pos, font_size=10, font_color="white", font_weight='bold')

    # Create legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                     markersize=10, label=f'Start (node {s})'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                     markersize=10, label=f'Goal (node {g})'))
    
    if decoy_endpoints:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                         markersize=10, label='Decoy endpoints'))
    
    # Add solution paths to legend
    for i, path in enumerate(solution_paths):
        color = path_colors[i % len(path_colors)]
        path_str = ' → '.join(map(str, path))
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                         label=f'Solution {i+1}: {path_str}'))
    
    # Add decoy paths to legend
    for i, path in enumerate(decoy_paths):
        color = decoy_colors[i % len(decoy_colors)]
        path_str = ' → '.join(map(str, path))
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2.5, linestyle='--',
                                         label=f'Decoy {i+1}: {path_str}'))
    
    legend_elements.append(plt.Line2D([0], [0], color='#CCCCCC', linewidth=1, 
                                     label='Other edges'))

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
    title_parts = [f'Graph {idx}: {g_nx.number_of_nodes()} nodes, {len(all_edges)} edges']
    if solution_paths:
        title_parts.append(f'{len(solution_paths)} solution paths')
    if decoy_paths:
        title_parts.append(f'{len(decoy_paths)} decoy paths')
    
    plt.title(', '.join(title_parts), fontsize=14, fontweight='bold')
    plt.axis("off")
    
    png_path = out_dir / f"{dataset_name}_sample_{idx:06d}.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize_graphs(data_dir: pathlib.Path, file_name: str, start_idx: int = 0, end_idx: int = None, indices: List[int] = None):
    """Visualize graphs from the saved dataset."""
    print(f"📊 Loading dataset from {data_dir}")
    
    # Extract dataset name from filename (remove .jsonl extension)
    dataset_name = pathlib.Path(file_name).stem
    
    # Load dataset
    records = load_dataset(data_dir, file_name)
    print(f"📈 Found {len(records)} graphs in dataset '{file_name}'")
    
    # Determine which indices to visualize
    if indices is not None:
        # Use specific indices
        indices = [idx for idx in indices if 0 <= idx < len(records)]
        if not indices:
            print("❌ No valid indices provided or all indices are out of range!")
            return
        indices.sort()
        print(f"🎯 Visualizing graphs at indices: {indices}")
        indices_to_process = indices
    else:
        # Use range-based approach
        if end_idx is None:
            end_idx = len(records)
        
        end_idx = min(end_idx, len(records))
        start_idx = max(0, start_idx)
        
        if start_idx >= end_idx:
            print(f"❌ Invalid range: start_idx ({start_idx}) >= end_idx ({end_idx})")
            return
        
        print(f"🎯 Visualizing graphs {start_idx} to {end_idx-1}")
        indices_to_process = list(range(start_idx, end_idx))
    
    # Create output directory for visualizations
    viz_dir = data_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Visualize each graph at the specified indices
    for i in indices_to_process:
        record = records[i]
        bundle = reconstruct_graph_from_record(record)
        
        decoy_info = f", {len(bundle.get('decoys', []))} decoy paths" if bundle.get('decoys') else ""
        print(f"🎨 Visualizing graph {i}: {len(bundle['graph'].edges())} edges, "
              f"start={bundle['start']}, goal={bundle['goal']}, "
              f"{len(bundle['paths'])} solution paths{decoy_info}")
        
        # Use the enhanced visualization function with dataset name
        enhanced_visualise_graph(bundle, viz_dir, i, dataset_name)
    
    print(f"✅ Visualization complete! Images saved to {viz_dir}")
    if indices is not None:
        print(f"🖼️  Generated {len(indices)} images for indices: {indices}")
        print(f"🏷️  Image names: {dataset_name}_sample_{{index:06d}}.png")
    else:
        print(f"🖼️  Generated {len(indices_to_process)} images: {dataset_name}_sample_{indices_to_process[0]:06d}.png to {dataset_name}_sample_{indices_to_process[-1]:06d}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved graphs from dataset.jsonl"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/path_samples",
        help="Directory containing the dataset.jsonl file"
    )
    
    # Create mutually exclusive group for range vs specific indices
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--indices",
        type=int,
        nargs='+',
        help="Specific indices to visualize (e.g., --indices 1 5 10 23)"
    )
    range_group.add_argument(
        "--range",
        action='store_true',
        help="Use range-based visualization (default if no --indices provided)"
    )
    
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Start index for range-based visualization (default: 0)"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=None,
        help="End index for range-based visualization (default: all graphs)"
    )
    
    parser.add_argument(
        "--file_name",
        type=str,
        default="dataset.jsonl",
        help="Name of the dataset file to visualize (default: dataset.jsonl)"
    )
    
    args = parser.parse_args()
    
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} does not exist!")
        sys.exit(1)
    
    try:
        if args.indices is not None:
            visualize_graphs(data_dir, args.file_name, indices=args.indices)
        else:
            visualize_graphs(data_dir, args.file_name, args.start_idx, args.end_idx)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 