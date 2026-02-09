import numpy as np
from scipy.optimize import linear_sum_assignment

def main():
    """
    Evaluate clustering accuracy using the Hungarian algorithm.
    
    Args:
        predicted_labels: List of predicted cluster IDs (0 to k-1)
        true_labels: List of true author IDs (0 to k-1)
        num_clusters: Number of clusters (k)
        
    Returns:
        accuracy: Fraction of correctly assigned chunks under optimal mapping
        mapping: Dictionary mapping predicted cluster ID to true author ID
    """
    predicted_labels = [0, 8, 8, 0, 2, 7, 4, 3, 1, 3, 8, 2, 0, 5, 7, 8, 6, 1, 7, 0, 7, 0, 9, 7, 8, 9, 5, 2, 1, 6, 4, 0, 1, 8, 6, 4, 2, 5, 6, 9, 0, 9, 0, 3, 3, 4, 7, 5, 0, 7, 2, 3, 6, 1, 5, 5, 5, 4, 8, 2, 0, 0, 8, 1, 7, 9, 9, 7, 6, 2, 8, 3, 2, 2, 5, 5, 3, 5, 4, 4, 8, 6, 7, 9, 5, 9, 8, 8, 8, 4, 5, 5, 2, 1, 6, 5, 9, 2, 0, 6, 6, 7, 3, 2, 4, 0, 0, 2, 1, 6, 6, 0, 2, 9, 6, 7, 9, 8, 0, 3, 6, 7, 6, 4, 6, 8, 3, 5, 4, 3, 3, 3, 9, 8, 1, 8, 7, 0, 3, 4, 6, 1, 4, 9, 6, 7, 0, 4, 4, 7, 3, 8, 4, 4, 8, 5, 3, 2, 8, 5, 3, 6, 8, 5, 5, 6, 2, 7, 5, 2, 3, 6, 5, 1, 2, 2, 4, 8, 8, 7, 1, 4, 6, 7, 2, 1, 9, 6, 0, 3, 0, 9, 9, 8, 6, 9, 2, 1, 8, 1, 0, 9, 9, 2, 6, 4, 4, 0, 3, 4, 8, 1, 8, 8, 8, 6, 0, 8, 7, 7, 3, 4, 9, 8, 6, 3, 4, 4, 2, 1, 4, 5, 1, 6, 5, 9, 8, 4, 5, 5, 5, 4, 3, 9, 7, 7, 6, 2, 1, 7, 8, 0, 1, 9, 9, 4, 5, 7, 9, 0, 2, 0, 7, 0, 3, 5, 2, 0, 9, 7, 1, 5, 5, 1, 5, 7, 1, 7, 7, 1, 3, 2, 5, 7, 6, 8, 1, 9, 1, 7, 3, 9, 8, 2, 5, 6, 5, 1, 3, 2, 8, 0, 5, 6, 4, 3, 0, 5, 5, 2, 9, 0, 7, 4, 9, 4, 5, 2, 2, 4, 1, 6, 3]
    true_labels = [3, 1, 1, 3, 0, 7, 4, 4, 6, 9, 1, 3, 3, 2, 7, 1, 5, 6, 4, 3, 7, 3, 8, 7, 1, 8, 2, 0, 6, 5, 4, 3, 6, 1, 5, 4, 0, 2, 5, 2, 3, 8, 3, 4, 9, 4, 7, 2, 3, 7, 0, 9, 5, 6, 2, 2, 2, 4, 1, 0, 3, 3, 1, 6, 7, 8, 8, 0, 5, 0, 1, 9, 0, 6, 3, 2, 9, 2, 4, 4, 1, 5, 7, 8, 2, 8, 1, 1, 1, 4, 8, 3, 0, 6, 5, 8, 8, 0, 3, 5, 0, 7, 9, 0, 4, 3, 3, 0, 6, 5, 0, 3, 0, 8, 9, 7, 8, 1, 3, 7, 5, 7, 5, 4, 5, 1, 9, 2, 4, 4, 9, 9, 8, 1, 0, 1, 7, 3, 9, 4, 5, 6, 4, 8, 5, 7, 3, 4, 4, 9, 9, 1, 4, 4, 1, 2, 9, 0, 1, 2, 9, 5, 1, 6, 2, 5, 0, 7, 6, 0, 9, 5, 5, 6, 0, 0, 4, 1, 1, 7, 6, 4, 5, 7, 0, 6, 8, 5, 8, 9, 3, 8, 8, 1, 5, 2, 0, 6, 1, 6, 3, 2, 3, 0, 5, 4, 4, 3, 7, 4, 1, 0, 1, 1, 1, 5, 3, 1, 7, 0, 9, 4, 8, 1, 5, 9, 4, 4, 0, 6, 4, 1, 6, 5, 6, 8, 1, 4, 2, 2, 2, 4, 9, 8, 7, 7, 0, 0, 6, 7, 1, 3, 6, 8, 2, 4, 2, 7, 8, 9, 0, 3, 7, 3, 9, 8, 5, 3, 8, 7, 6, 2, 2, 6, 2, 7, 6, 7, 7, 6, 9, 0, 2, 7, 5, 1, 2, 8, 3, 7, 9, 8, 1, 0, 2, 5, 2, 6, 4, 0, 1, 3, 2, 5, 9, 9, 3, 2, 8, 0, 8, 3, 7, 4, 8, 4, 8, 1, 5, 4, 6, 5, 9]
    num_clusters = 10

    # Create confusion matrix
    confusion_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    
    for p, t in zip(predicted_labels, true_labels):
        if 0 <= p < num_clusters and 0 <= t < num_clusters:
            confusion_matrix[p][t] += 1
            
    # We want to maximize correct assignments, which is equivalent to 
    # finding a permutation that maximizes sum of diagonal elements after permutation.
    # linear_sum_assignment minimizes cost, so we pass negative confusion matrix.
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # Calculate accuracy
    total_correct = confusion_matrix[row_ind, col_ind].sum()
    total_samples = len(predicted_labels)
    accuracy = total_correct / total_samples
    
    # Mapping from Predicted (row) -> True (col)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    
    print(accuracy, mapping)

    return accuracy, mapping

if __name__=="__main__":
    main()