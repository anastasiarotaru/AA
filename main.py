import time
import random
import sys
import matplotlib.pyplot as plt

# Increase recursion depth for QuickSort on sorted/reversed lists
sys.setrecursionlimit(10000)

# --- 1. ALGORITHM IMPLEMENTATIONS ---

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    # Using middle element as pivot to handle sorted/reversed data better
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def heapify(arr, n, i):
    largest = i
    l, r = 2 * i + 1, 2 * i + 2
    if l < n and arr[i] < arr[l]: largest = l
    if r < n and arr[largest] < arr[r]: largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

# --- 2. EMPIRICAL ANALYSIS SETUP ---

sizes = [100, 500, 1000, 2000, 4000, 6000]
algorithms = {
    "QuickSort": quick_sort,
    "MergeSort": merge_sort,
    "HeapSort": heap_sort,
    "ShellSort": shell_sort
}

results = {name: [] for name in algorithms}

print(f"{'Algorithm':<12} | {'Size':<6} | {'Time (ms)':<10}")
print("-" * 35)

for size in sizes:
    # Generate Random Data (Task 2: Properties of input)
    test_data = [random.randint(0, 100000) for _ in range(size)]
    
    for name, func in algorithms.items():
        # Copy data to ensure same starting point for each algorithm
        data_copy = list(test_data)
        
        start_time = time.perf_counter()
        func(data_copy)
        end_time = time.perf_counter()
        
        # Convert to milliseconds (Task 3: Comparison metric)
        elapsed_ms = (end_time - start_time) * 1000
        results[name].append(elapsed_ms)
        
        print(f"{name:<12} | {size:<6} | {elapsed_ms:.4f}")

# --- 3. GRAPH GENERATOR (Task 5: Graphical Presentation) ---

plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(sizes, times, marker='o', label=name)



plt.title('Empirical Analysis of Sorting Algorithms (Random Data)')
plt.xlabel('Array Size (N)')
plt.ylabel('Execution Time (ms)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the plot for the Overleaf report
plt.savefig('plot_sorting.png')
print("\nGraph saved as 'plot_sorting.png'")
plt.show()