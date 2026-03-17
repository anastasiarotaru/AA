"""
Empirical Analysis of Depth-First Search (DFS) and Breadth-First Search (BFS)
Author: Anastasia Rotaru
FAF-242, Technical University of Moldova

This module implements and analyzes DFS and BFS algorithms on various graph structures.
Includes bonus features: multiple graph representations, visualization, and comprehensive metrics.
"""

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import psutil
import os
import csv
from datetime import datetime

# ============================================================================
# BONUS FEATURE 1: Multiple Graph Representations
# ============================================================================

class Graph:
    """
    A graph class that supports multiple internal representations.
    This demonstrates understanding of different data structures for graph storage.
    """
    
    def __init__(self, vertices, representation='adjacency_list'):
        """
        Initialize a graph with a specified number of vertices and representation.
        
        Args:
            vertices: Number of vertices in the graph
            representation: Type of representation ('adjacency_list', 'adjacency_matrix', 'edge_list')
        """
        self.V = vertices
        self.representation = representation
        self.repr_name = representation.replace('_', ' ').title()
        
        # Initialize based on representation type
        if representation == 'adjacency_list':
            self.adj = [[] for _ in range(vertices)]
            self.storage_type = "List of Lists"
            
        elif representation == 'adjacency_matrix':
            self.adj = [[0] * vertices for _ in range(vertices)]
            self.storage_type = "2D Matrix"
            
        elif representation == 'edge_list':
            self.adj = []  # Will store edges as tuples (u, v)
            self.storage_type = "List of Edges"
            
        else:
            raise ValueError(f"Unknown representation: {representation}")
    
    def add_edge(self, u, v):
        """Add an undirected edge between vertices u and v."""
        if u >= self.V or v >= self.V:
            raise ValueError(f"Vertex index out of range. Max vertex: {self.V-1}")
        
        if self.representation == 'adjacency_list':
            self.adj[u].append(v)
            self.adj[v].append(u)
            
        elif self.representation == 'adjacency_matrix':
            self.adj[u][v] = 1
            self.adj[v][u] = 1
            
        elif self.representation == 'edge_list':
            self.adj.append((u, v))
            # For undirected, we might want to store both directions or handle carefully
            # For simplicity, we'll store each edge once and handle traversal differently
    
    def get_neighbors(self, vertex):
        """Return all neighbors of a given vertex."""
        if self.representation == 'adjacency_list':
            return self.adj[vertex]
            
        elif self.representation == 'adjacency_matrix':
            neighbors = []
            for i in range(self.V):
                if self.adj[vertex][i] == 1:
                    neighbors.append(i)
            return neighbors
            
        elif self.representation == 'edge_list':
            neighbors = []
            for u, v in self.adj:
                if u == vertex:
                    neighbors.append(v)
                elif v == vertex:
                    neighbors.append(u)
            return neighbors
    
    def get_storage_size(self):
        """Calculate approximate memory usage of the graph representation in bytes."""
        import sys
        
        if self.representation == 'adjacency_list':
            # Each list overhead + integers
            size = sys.getsizeof(self.adj)
            for lst in self.adj:
                size += sys.getsizeof(lst)
                size += len(lst) * sys.getsizeof(0)  # Each integer
            return size
            
        elif self.representation == 'adjacency_matrix':
            # Matrix of integers
            size = sys.getsizeof(self.adj)
            for row in self.adj:
                size += sys.getsizeof(row)
                size += len(row) * sys.getsizeof(0)
            return size
            
        elif self.representation == 'edge_list':
            # List of tuples
            size = sys.getsizeof(self.adj)
            for edge in self.adj:
                size += sys.getsizeof(edge)
                size += 2 * sys.getsizeof(0)
            return size
    
    def __str__(self):
        """String representation of the graph."""
        if self.representation == 'adjacency_list':
            result = f"Graph with {self.V} vertices (Adjacency List):\n"
            for i in range(min(self.V, 10)):  # Show first 10 vertices
                result += f"  {i}: {self.adj[i][:5]}{'...' if len(self.adj[i]) > 5 else ''}\n"
            if self.V > 10:
                result += "  ...\n"
            return result
            
        elif self.representation == 'adjacency_matrix':
            result = f"Graph with {self.V} vertices (Adjacency Matrix - first 10x10):\n"
            for i in range(min(self.V, 10)):
                result += "  " + "".join(str(self.adj[i][j]) for j in range(min(self.V, 10))) + "\n"
            return result
            
        else:
            return f"Graph with {self.V} vertices ({self.repr_name}), {len(self.adj)} edges"


# ============================================================================
# BONUS FEATURE 2: Graph Generator with Various Topologies
# ============================================================================

class GraphGenerator:
    """Generate graphs with different properties for empirical analysis."""
    
    @staticmethod
    def generate_random_graph(vertices, edge_density, seed=None):
        """
        Generate a random graph with given vertex count and edge density.
        
        Args:
            vertices: Number of vertices
            edge_density: Value between 0 and 1 representing edge probability
            seed: Random seed for reproducibility
        
        Returns:
            Graph object with random edges
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate number of edges based on density
        max_edges = vertices * (vertices - 1) // 2
        target_edges = int(max_edges * edge_density)
        
        # Create graph
        g = Graph(vertices, 'adjacency_list')
        
        # Generate random edges without duplicates
        edges_added = 0
        attempts = 0
        max_attempts = target_edges * 10
        
        while edges_added < target_edges and attempts < max_attempts:
            u = np.random.randint(0, vertices)
            v = np.random.randint(0, vertices)
            if u != v:
                # Check if edge already exists (simplified - we'll just add and hope for low duplicates)
                if v not in g.adj[u]:  # This check is O(degree), fine for moderate graphs
                    g.add_edge(u, v)
                    edges_added += 1
            attempts += 1
        
        return g
    
    @staticmethod
    def generate_sparse_graph(vertices, edges_per_vertex=2):
        """Generate a sparse graph with approximately edges_per_vertex * vertices edges."""
        density = (edges_per_vertex * vertices) / (vertices * (vertices - 1) / 2)
        density = min(density, 0.5)  # Cap at 0.5 to ensure sparsity
        return GraphGenerator.generate_random_graph(vertices, density)
    
    @staticmethod
    def generate_dense_graph(vertices, density_factor=0.25):
        """Generate a dense graph with given density factor (default 25% of max edges)."""
        return GraphGenerator.generate_random_graph(vertices, density_factor)
    
    @staticmethod
    def generate_path_graph(vertices):
        """Generate a path graph where each vertex connects to its neighbors."""
        g = Graph(vertices, 'adjacency_list')
        for i in range(vertices - 1):
            g.add_edge(i, i + 1)
        return g
    
    @staticmethod
    def generate_star_graph(vertices):
        """Generate a star graph with center at vertex 0."""
        g = Graph(vertices, 'adjacency_list')
        for i in range(1, vertices):
            g.add_edge(0, i)
        return g
    
    @staticmethod
    def generate_complete_graph(vertices):
        """Generate a complete graph where every vertex connects to every other."""
        g = Graph(vertices, 'adjacency_list')
        for i in range(vertices):
            for j in range(i + 1, vertices):
                g.add_edge(i, j)
        return g


# ============================================================================
# Algorithm Implementations
# ============================================================================

class GraphTraversal:
    """Class containing DFS and BFS traversal algorithms with performance tracking."""
    
    def __init__(self, graph):
        """
        Initialize with a graph to traverse.
        
        Args:
            graph: Graph object to perform traversals on
        """
        self.graph = graph
        self.V = graph.V
    
    def bfs(self, start_vertex=0, track_metrics=True):
        """
        Breadth-First Search traversal.
        
        Args:
            start_vertex: Vertex to start traversal from
            track_metrics: Whether to track memory usage metrics
        
        Returns:
            Tuple of (traversal_order, metrics_dict)
        """
        visited = [False] * self.V
        queue = deque()
        traversal = []
        parent = [-1] * self.V  # For path reconstruction
        distance = [-1] * self.V  # Distance from start
        
        # Metrics tracking
        max_queue_size = 0
        operations = 0
        
        # Start traversal
        visited[start_vertex] = True
        queue.append(start_vertex)
        distance[start_vertex] = 0
        max_queue_size = max(max_queue_size, len(queue))
        
        while queue:
            vertex = queue.popleft()
            operations += 1
            traversal.append(vertex)
            
            # Get neighbors based on graph representation
            neighbors = self.graph.get_neighbors(vertex)
            
            for neighbour in neighbors:
                operations += 1
                if not visited[neighbour]:
                    visited[neighbour] = True
                    parent[neighbour] = vertex
                    distance[neighbour] = distance[vertex] + 1
                    queue.append(neighbour)
                    max_queue_size = max(max_queue_size, len(queue))
        
        metrics = {
            'max_queue_size': max_queue_size,
            'total_operations': operations,
            'vertices_visited': len(traversal),
            'avg_queue_size': sum(len(queue) for _ in range(len(traversal))) / len(traversal) if traversal else 0
        }
        
        if track_metrics:
            return traversal, metrics, {'parent': parent, 'distance': distance}
        return traversal
    
    def dfs(self, start_vertex=0, track_metrics=True):
        """
        Iterative Depth-First Search traversal.
        
        Args:
            start_vertex: Vertex to start traversal from
            track_metrics: Whether to track memory usage metrics
        
        Returns:
            Tuple of (traversal_order, metrics_dict)
        """
        visited = [False] * self.V
        stack = []
        traversal = []
        parent = [-1] * self.V
        
        # Metrics tracking
        max_stack_size = 0
        operations = 0
        
        # Start traversal
        stack.append(start_vertex)
        max_stack_size = max(max_stack_size, len(stack))
        
        while stack:
            vertex = stack.pop()
            operations += 1
            
            if not visited[vertex]:
                visited[vertex] = True
                traversal.append(vertex)
                
                # Get neighbors (reverse order for more natural DFS order)
                neighbors = self.graph.get_neighbors(vertex)
                # Reverse to simulate recursive order (optional)
                for neighbour in reversed(neighbors):
                    operations += 1
                    if not visited[neighbour]:
                        parent[neighbour] = vertex
                        stack.append(neighbour)
                        max_stack_size = max(max_stack_size, len(stack))
        
        metrics = {
            'max_stack_size': max_stack_size,
            'total_operations': operations,
            'vertices_visited': len(traversal),
            'avg_stack_size': sum(len(stack) for _ in range(len(traversal))) / len(traversal) if traversal else 0
        }
        
        if track_metrics:
            return traversal, metrics, {'parent': parent}
        return traversal
    
    def compare_traversals(self, start_vertex=0):
        """
        Compare BFS and DFS on the same graph.
        
        Returns:
            Dictionary with comparison metrics
        """
        # Run BFS
        bfs_start = time.perf_counter()
        bfs_order, bfs_metrics, bfs_extra = self.bfs(start_vertex, track_metrics=True)
        bfs_time = (time.perf_counter() - bfs_start) * 1000  # Convert to ms
        
        # Run DFS
        dfs_start = time.perf_counter()
        dfs_order, dfs_metrics, dfs_extra = self.dfs(start_vertex, track_metrics=True)
        dfs_time = (time.perf_counter() - dfs_start) * 1000  # Convert to ms
        
        # Calculate additional metrics
        comparison = {
            'bfs': {
                'time_ms': bfs_time,
                'max_memory': bfs_metrics['max_queue_size'],
                'operations': bfs_metrics['total_operations'],
                'traversal': bfs_order[:20]  # First 20 nodes for preview
            },
            'dfs': {
                'time_ms': dfs_time,
                'max_memory': dfs_metrics['max_stack_size'],
                'operations': dfs_metrics['total_operations'],
                'traversal': dfs_order[:20]
            },
            'differences': {
                'time_diff_ms': abs(bfs_time - dfs_time),
                'memory_diff': abs(bfs_metrics['max_queue_size'] - dfs_metrics['max_stack_size']),
                'faster': 'BFS' if bfs_time < dfs_time else 'DFS',
                'memory_efficient': 'BFS' if bfs_metrics['max_queue_size'] < dfs_metrics['max_stack_size'] else 'DFS'
            }
        }
        
        return comparison


# ============================================================================
# BONUS FEATURE 3: Performance Visualization
# ============================================================================

class PerformanceAnalyzer:
    """Class for analyzing and visualizing algorithm performance."""
    
    def __init__(self):
        self.results = {
            'sparse': {'sizes': [], 'bfs_times': [], 'dfs_times': [], 
                      'bfs_memory': [], 'dfs_memory': []},
            'dense': {'sizes': [], 'bfs_times': [], 'dfs_times': [], 
                     'bfs_memory': [], 'dfs_memory': []},
            'path': {'sizes': [], 'bfs_times': [], 'dfs_times': [], 
                    'bfs_memory': [], 'dfs_memory': []},
            'star': {'sizes': [], 'bfs_times': [], 'dfs_times': [], 
                    'bfs_memory': [], 'dfs_memory': []}
        }
    
    def run_benchmark(self, sizes=[100, 500, 1000, 2500, 5000], runs_per_size=5):
        """
        Run comprehensive benchmark on various graph types and sizes.
        
        Args:
            sizes: List of graph sizes to test
            runs_per_size: Number of runs per configuration (for averaging)
        """
        print("=" * 70)
        print("DFS vs BFS EMPIRICAL ANALYSIS BENCHMARK")
        print("=" * 70)
        print(f"Testing graph sizes: {sizes}")
        print(f"Runs per configuration: {runs_per_size}")
        print("-" * 70)
        
        for size in sizes:
            print(f"\n--- Testing with {size} vertices ---")
            
            # Test sparse graph
            sparse_times_bfs = []
            sparse_times_dfs = []
            sparse_mem_bfs = []
            sparse_mem_dfs = []
            
            for run in range(runs_per_size):
                g = GraphGenerator.generate_sparse_graph(size, edges_per_vertex=2)
                trav = GraphTraversal(g)
                
                # BFS
                start = time.perf_counter()
                _, bfs_metrics, _ = trav.bfs(track_metrics=True)
                sparse_times_bfs.append((time.perf_counter() - start) * 1000)
                sparse_mem_bfs.append(bfs_metrics['max_queue_size'])
                
                # DFS
                start = time.perf_counter()
                _, dfs_metrics, _ = trav.dfs(track_metrics=True)
                sparse_times_dfs.append((time.perf_counter() - start) * 1000)
                sparse_mem_dfs.append(dfs_metrics['max_stack_size'])
            
            self.results['sparse']['sizes'].append(size)
            self.results['sparse']['bfs_times'].append(np.mean(sparse_times_bfs))
            self.results['sparse']['dfs_times'].append(np.mean(sparse_times_dfs))
            self.results['sparse']['bfs_memory'].append(np.mean(sparse_mem_bfs))
            self.results['sparse']['dfs_memory'].append(np.mean(sparse_mem_dfs))
            
            print(f"  Sparse Graph - BFS: {self.results['sparse']['bfs_times'][-1]:.2f}ms, "
                  f"DFS: {self.results['sparse']['dfs_times'][-1]:.2f}ms")
            
            # Test dense graph
            dense_times_bfs = []
            dense_times_dfs = []
            dense_mem_bfs = []
            dense_mem_dfs = []
            
            for run in range(runs_per_size):
                g = GraphGenerator.generate_dense_graph(size, density_factor=0.25)
                trav = GraphTraversal(g)
                
                start = time.perf_counter()
                _, bfs_metrics, _ = trav.bfs(track_metrics=True)
                dense_times_bfs.append((time.perf_counter() - start) * 1000)
                dense_mem_bfs.append(bfs_metrics['max_queue_size'])
                
                start = time.perf_counter()
                _, dfs_metrics, _ = trav.dfs(track_metrics=True)
                dense_times_dfs.append((time.perf_counter() - start) * 1000)
                dense_mem_dfs.append(dfs_metrics['max_stack_size'])
            
            self.results['dense']['sizes'].append(size)
            self.results['dense']['bfs_times'].append(np.mean(dense_times_bfs))
            self.results['dense']['dfs_times'].append(np.mean(dense_times_dfs))
            self.results['dense']['bfs_memory'].append(np.mean(dense_mem_bfs))
            self.results['dense']['dfs_memory'].append(np.mean(dense_mem_dfs))
            
            print(f"  Dense Graph  - BFS: {self.results['dense']['bfs_times'][-1]:.2f}ms, "
                  f"DFS: {self.results['dense']['dfs_times'][-1]:.2f}ms")
            
            # Test special topologies
            for graph_type in ['path', 'star']:
                times_bfs = []
                times_dfs = []
                mem_bfs = []
                mem_dfs = []
                
                for run in range(runs_per_size):
                    if graph_type == 'path':
                        g = GraphGenerator.generate_path_graph(size)
                    else:
                        g = GraphGenerator.generate_star_graph(size)
                    
                    trav = GraphTraversal(g)
                    
                    start = time.perf_counter()
                    _, bfs_metrics, _ = trav.bfs(track_metrics=True)
                    times_bfs.append((time.perf_counter() - start) * 1000)
                    mem_bfs.append(bfs_metrics['max_queue_size'])
                    
                    start = time.perf_counter()
                    _, dfs_metrics, _ = trav.dfs(track_metrics=True)
                    times_dfs.append((time.perf_counter() - start) * 1000)
                    mem_dfs.append(dfs_metrics['max_stack_size'])
                
                self.results[graph_type]['sizes'].append(size)
                self.results[graph_type]['bfs_times'].append(np.mean(times_bfs))
                self.results[graph_type]['dfs_times'].append(np.mean(times_dfs))
                self.results[graph_type]['bfs_memory'].append(np.mean(mem_bfs))
                self.results[graph_type]['dfs_memory'].append(np.mean(mem_dfs))
    
    def plot_results(self, save_figures=True):
        """
        Generate comprehensive visualization plots.
        
        Args:
            save_figures: Whether to save figures to disk
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = {'bfs': '#1f77b4', 'dfs': '#ff7f0e'}
        
        # Figure 1: Execution Time Comparison for Different Graph Types
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DFS vs BFS Execution Time Comparison', fontsize=16, fontweight='bold')
        
        graph_types = ['sparse', 'dense', 'path', 'star']
        titles = ['Sparse Graphs (E ≈ 2V)', 'Dense Graphs (25% Density)', 
                  'Path Graphs (Linear)', 'Star Graphs (Hub & Spokes)']
        
        for idx, (graph_type, title) in enumerate(zip(graph_types, titles)):
            ax = axes[idx // 2, idx % 2]
            
            sizes = self.results[graph_type]['sizes']
            bfs_times = self.results[graph_type]['bfs_times']
            dfs_times = self.results[graph_type]['dfs_times']
            
            ax.plot(sizes, bfs_times, 'o-', color=colors['bfs'], linewidth=2, markersize=8, label='BFS')
            ax.plot(sizes, dfs_times, 's-', color=colors['dfs'], linewidth=2, markersize=8, label='DFS')
            
            ax.set_xlabel('Number of Vertices (V)', fontsize=11)
            ax.set_ylabel('Execution Time (ms)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig('bfs_dfs_time_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Memory Usage Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DFS vs BFS Memory Usage Comparison', fontsize=16, fontweight='bold')
        
        for idx, (graph_type, title) in enumerate(zip(graph_types, titles)):
            ax = axes[idx // 2, idx % 2]
            
            sizes = self.results[graph_type]['sizes']
            bfs_mem = self.results[graph_type]['bfs_memory']
            dfs_mem = self.results[graph_type]['dfs_memory']
            
            ax.plot(sizes, bfs_mem, 'o-', color=colors['bfs'], linewidth=2, markersize=8, label='BFS Queue')
            ax.plot(sizes, dfs_mem, 's-', color=colors['dfs'], linewidth=2, markersize=8, label='DFS Stack')
            
            ax.set_xlabel('Number of Vertices (V)', fontsize=11)
            ax.set_ylabel('Peak Memory Usage (nodes in queue/stack)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig('bfs_dfs_memory_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Figure 3: Time Ratio Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for graph_type in graph_types:
            sizes = self.results[graph_type]['sizes']
            bfs_times = self.results[graph_type]['bfs_times']
            dfs_times = self.results[graph_type]['dfs_times']
            
            # Calculate BFS/DFS time ratio
            ratio = [b/d if d > 0 else 1 for b, d in zip(bfs_times, dfs_times)]
            ax.plot(sizes, ratio, 'o-', linewidth=2, markersize=8, label=graph_type.capitalize())
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        ax.set_xlabel('Number of Vertices (V)', fontsize=12)
        ax.set_ylabel('BFS Time / DFS Time Ratio', fontsize=12)
        ax.set_title('Performance Ratio: BFS vs DFS (Ratio > 1 means BFS slower)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_figures:
            plt.savefig('bfs_dfs_ratio_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def export_results_to_csv(self, filename='benchmark_results.csv'):
        """Export benchmark results to CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Graph Type', 'Vertices', 'BFS Time (ms)', 'DFS Time (ms)', 
                           'BFS Memory', 'DFS Memory'])
            
            for graph_type in self.results:
                for i, size in enumerate(self.results[graph_type]['sizes']):
                    writer.writerow([
                        graph_type,
                        size,
                        f"{self.results[graph_type]['bfs_times'][i]:.2f}",
                        f"{self.results[graph_type]['dfs_times'][i]:.2f}",
                        self.results[graph_type]['bfs_memory'][i],
                        self.results[graph_type]['dfs_memory'][i]
                    ])
        
        print(f"\nResults exported to {filename}")


# ============================================================================
# BONUS FEATURE 4: Graph Representation Comparison
# ============================================================================

def compare_representations():
    """Compare different graph representations in terms of memory and access time."""
    print("\n" + "=" * 70)
    print("BONUS: Graph Representation Comparison")
    print("=" * 70)
    
    sizes = [100, 500, 1000]
    representations = ['adjacency_list', 'adjacency_matrix', 'edge_list']
    
    results = []
    
    for size in sizes:
        print(f"\n--- Testing with {size} vertices ---")
        
        for rep in representations:
            # Create graph with this representation
            g = Graph(size, representation=rep)
            
            # Add some random edges (sparse graph)
            edge_count = 0
            for _ in range(size * 2):  # Add ~2*V edges
                u = np.random.randint(0, size)
                v = np.random.randint(0, size)
                if u != v:
                    try:
                        g.add_edge(u, v)
                        edge_count += 1
                    except:
                        pass
            
            # Measure storage size
            storage_bytes = g.get_storage_size()
            storage_kb = storage_bytes / 1024
            
            # Measure neighbor access time
            trav = GraphTraversal(g)
            start = time.perf_counter()
            for i in range(min(100, size)):
                _ = g.get_neighbors(i)
            access_time = (time.perf_counter() - start) * 1000  # ms
            
            results.append({
                'size': size,
                'representation': g.repr_name,
                'edges': edge_count,
                'storage_kb': storage_kb,
                'access_time_ms': access_time
            })
            
            print(f"  {g.repr_name:20s} | Edges: {edge_count:5d} | "
                  f"Storage: {storage_kb:8.2f} KB | Access: {access_time:6.3f} ms")
    
    return results


# ============================================================================
# BONUS FEATURE 5: Interactive Demonstration
# ============================================================================

def interactive_demo():
    """Run an interactive demonstration of DFS and BFS on a small graph."""
    print("\n" + "=" * 70)
    print("INTERACTIVE DFS & BFS DEMONSTRATION")
    print("=" * 70)
    
    # Create a small example graph
    g = Graph(7, 'adjacency_list')
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for u, v in edges:
        g.add_edge(u, v)
    
    print("\nExample Graph Structure (7 vertices):")
    print("   0")
    print("  / \\")
    print(" 1   2")
    print(" /\\   /\\")
    print("3 4 5 6")
    
    trav = GraphTraversal(g)
    
    # Run BFS
    print("\n" + "-" * 40)
    print("BFS TRAVERSAL (starting from vertex 0)")
    print("-" * 40)
    
    bfs_order, bfs_metrics, bfs_extra = trav.bfs(0, track_metrics=True)
    print(f"Traversal Order: {' -> '.join(map(str, bfs_order))}")
    print(f"Distances from start: {bfs_extra['distance']}")
    print(f"Max Queue Size: {bfs_metrics['max_queue_size']}")
    print(f"Total Operations: {bfs_metrics['total_operations']}")
    
    # Run DFS
    print("\n" + "-" * 40)
    print("DFS TRAVERSAL (starting from vertex 0)")
    print("-" * 40)
    
    dfs_order, dfs_metrics, dfs_extra = trav.dfs(0, track_metrics=True)
    print(f"Traversal Order: {' -> '.join(map(str, dfs_order))}")
    print(f"Max Stack Size: {dfs_metrics['max_stack_size']}")
    print(f"Total Operations: {dfs_metrics['total_operations']}")
    
    # Compare
    print("\n" + "-" * 40)
    print("COMPARISON SUMMARY")
    print("-" * 40)
    comparison = trav.compare_traversals(0)
    
    print(f"BFS Time: {comparison['bfs']['time_ms']:.3f} ms")
    print(f"DFS Time: {comparison['dfs']['time_ms']:.3f} ms")
    print(f"BFS Max Memory: {comparison['bfs']['max_memory']} nodes")
    print(f"DFS Max Memory: {comparison['dfs']['max_memory']} nodes")
    print(f"Faster Algorithm: {comparison['differences']['faster']}")
    print(f"More Memory Efficient: {comparison['differences']['memory_efficient']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EMPIRICAL ANALYSIS OF DFS AND BFS ALGORITHMS")
    print("=" * 70)
    print("Author: Anastasia Rotaru, FAF-242")
    print("Technical University of Moldova")
    print("=" * 70)
    
    # Part 1: Interactive Demo on small graph
    interactive_demo()
    
    # Part 2: Graph Representation Comparison (BONUS)
    input("\nPress Enter to continue to Graph Representation Comparison...")
    rep_results = compare_representations()
    
    # Part 3: Full Benchmark
    input("\nPress Enter to continue to Full Performance Benchmark...")
    
    analyzer = PerformanceAnalyzer()
    
    # Run benchmark with smaller sizes for demo (adjust for full analysis)
    # For full analysis, use: sizes=[100, 500, 1000, 2500, 5000]
    analyzer.run_benchmark(sizes=[100, 500, 1000], runs_per_size=3)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    analyzer.plot_results(save_figures=True)
    
    # Export results
    analyzer.export_results_to_csv('bfs_dfs_benchmark_results.csv')
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. bfs_dfs_time_comparison.png - Time performance graphs")
    print("  2. bfs_dfs_memory_comparison.png - Memory usage graphs")
    print("  3. bfs_dfs_ratio_comparison.png - Performance ratio analysis")
    print("  4. bfs_dfs_benchmark_results.csv - Raw data for further analysis")
    print("\n" + "=" * 70)