import time
import numpy as np
from app.services.search_service import SimilaritySearch

def benchmark_search(index_size=10000, queries=100):
    """Benchmark search latency"""
    dim = 2048
    index = SimilaritySearch(dimension=dim)
    
    # Generate random vectors
    vectors = np.random.random((index_size, dim)).astype('float32')
    # Normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    metadata = [{'id': i} for i in range(index_size)]
    index.add_vectors(vectors, metadata)
    
    # Benchmark
    query = np.random.random((1, dim)).astype('float32')
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    
    latencies = []
    for _ in range(queries):
        start = time.time()
        index.search(query, k=3)
        latencies.append((time.time() - start) * 1000)  # ms
    
    print(f"Index size: {index_size}")
    print(f"Mean latency: {np.mean(latencies):.2f}ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f}ms")

if __name__ == "__main__":
    benchmark_search()
