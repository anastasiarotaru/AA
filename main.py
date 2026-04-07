import time
import heapq
import random
import matplotlib.pyplot as plt

def dijkstra(adj_list, start_node):
    n = len(adj_list)
    distances = [float('inf')] * n
    distances[start_node] = 0
    pq = [(0, start_node)]
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]: continue
        for v, weight in adj_list[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))
    return distances

def floyd_warshall(matrix):
    n = len(matrix)
    dist = [row[:] for row in matrix]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def get_graphs(n, density):
    adj_list = [[] for _ in range(n)]
    matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n): matrix[i][i] = 0
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                w = random.randint(1, 50)
                adj_list[i].append((j, w)); adj_list[j].append((i, w))
                matrix[i][j] = matrix[j][i] = w
    return adj_list, matrix

# --- BENCHMARKING ---
nodes_range = [10, 30, 50, 80, 100, 150]
results = []

print(f"{'V':<5} | {'Sparse Dijk':<12} | {'Sparse FW':<10} | {'Dense Dijk':<12} | {'Dense FW':<10}")
print("-" * 60)

plot_data = {"ds": [], "fs": [], "dd": [], "fd": []}

for n in nodes_range:
    # Sparse
    s_list, s_mat = get_graphs(n, 0.1)
    t1 = time.time(); [dijkstra(s_list, i) for i in range(n)]; d_sparse = time.time() - t1
    t2 = time.time(); floyd_warshall(s_mat); f_sparse = time.time() - t2
    
    # Dense
    d_list, d_mat = get_graphs(n, 0.8)
    t3 = time.time(); [dijkstra(d_list, i) for i in range(n)]; d_dense = time.time() - t3
    t4 = time.time(); floyd_warshall(d_mat); f_dense = time.time() - t4

    print(f"{n:<5} | {d_sparse:<12.5f} | {f_sparse:<10.5f} | {d_dense:<12.5f} | {f_dense:<10.5f}")
    
    plot_data["ds"].append(d_sparse); plot_data["fs"].append(f_sparse)
    plot_data["dd"].append(d_dense); plot_data["fd"].append(f_dense)

# --- PLOTTING ---
plt.figure(figsize=(10, 5))
plt.plot(nodes_range, plot_data["fs"], 'r-x', label='Floyd-Warshall (Sparse)')
plt.plot(nodes_range, plot_data["ds"], 'b-o', label='Dijkstra All-Pairs (Sparse)')
plt.title("Execution Time Comparison")
plt.xlabel("Number of Nodes")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)
plt.show()