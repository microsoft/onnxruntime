import sys
import numpy as np
from numpy.linalg import norm

def read_values(filename):
    with open(filename, 'r') as file:
        values = np.array([float(line.strip()) for line in file])
    return values

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cosine_similarity.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    vec1 = read_values(file1)
    vec2 = read_values(file2)

    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine Similarity: {similarity}")
