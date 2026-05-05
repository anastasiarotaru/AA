import heapq
import time
import random
import matplotlib.pyplot as plt

# --- ALGORITHM IMPLEMENTATIONS ---

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                self.rank[root_j] += 1
            return True
        return False

def kruskal_mst(n, edge_list):
    start_time = time.perf_counter()
    edge_list.sort(key=lambda x: x[2])
    uf = UnionFind(n)
    edges_count = 0
    for u, v, w in edge_list:
        if uf.union(u, v):
            edges_count += 1
            if edges_count == n - 1: break
    return (time.perf_counter() - start_time) * 1000  # Convert to ms

def prim_mst(n, adj_list):
    start_time = time.perf_counter()
    visited = [False] * n
    pq = [(0, 0)]
    nodes_included = 0
    while pq and nodes_included < n:
        w, u = heapq.heappop(pq)
        if visited[u]: continue
        visited[u] = True
        nodes_included += 1
        for v, weight in adj_list[u]:
            if not visited[v]:
                heapq.heappush(pq, (weight, v))
    return (time.perf_counter() - start_time) * 1000  # Convert to ms

# --- DATA GENERATION ---

def generate_test_data(n, density):
    adj_list = [[] for _ in range(n)]
    edge_list = []
    for i in range(n - 1):
        w = random.randint(1, 100)
        adj_list[i].append((i + 1, w)); adj_list[i + 1].append((i, w))
        edge_list.append((i, i + 1, w))
    max_edges = n * (n - 1) // 2
    target_edges = int(max_edges * density)
    current_edges = n - 1
    while current_edges < target_edges:
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u != v:
            w = random.randint(1, 100)
            adj_list[u].append((v, w)); adj_list[v].append((u, w))
            edge_list.append((u, v, w))
            current_edges += 1
    return adj_list, edge_list

# --- MAIN BENCHMARK & PLOTTING ---

if __name__ == "__main__":
    node_counts = [50, 100, 200, 300, 400, 500]
    prim_times, kruskal_times = [], []

    print(f"{'V':<5} | {'Prim (Dense) ms':<15} | {'Kruskal (Dense) ms':<15}")
    print("-" * 40)

    for v in node_counts:
        # We test on Dense graphs (density 0.8) to see the performance split clearly
        adj, edges = generate_test_data(v, 0.8)
        
        t_p = prim_mst(v, adj)
        t_k = kruskal_mst(v, edges)
        
        prim_times.append(t_p)
        kruskal_times.append(t_k)
        print(f"{v:<5} | {t_p:15.4f} | {t_k:15.4f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, prim_times, label="Prim's Algorithm", marker='o', linestyle='-', linewidth=2)
    plt.plot(node_counts, kruskal_times, label="Kruskal's Algorithm", marker='s', linestyle='--', linewidth=2)
    
    plt.title('MST Algorithm Performance Scaling (Dense Graphs)', fontsize=14)
    plt.xlabel('Number of Vertices (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot for Overleaf
    plt.savefig('mst_performance.png')
    print("\nGraph saved as 'mst_performance.png'. Upload this to Overleaf!")
    plt.show()