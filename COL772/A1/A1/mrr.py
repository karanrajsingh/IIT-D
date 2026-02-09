import json
import sys

def calculate_mrr(ranked_file):
    """Calculate MRR from ranked candidates JSONL"""
    total_queries = 0
    total_reciprocal_rank = 0.0
    
    with open(ranked_file, 'r') as f:
        for line in f:
            query_data = json.loads(line.strip())
            query_id = query_data["query_id"]  # "q_0", "q_1", etc.
            ranked_list = query_data["ranked_candidates"]  # ["cand_6", "cand_5", ...]
            
            # Correct answer: q_0 → cand_1, q_1 → cand_2, q_2 → cand_3, etc.
            query_num = int(query_id.split('_')[1])  # 0, 1, 2...
            correct_cand = f"cand_{query_num%10 + 1}"    # cand_1, cand_2, cand_3...
            
            # Find position of correct candidate (1-indexed)
            if correct_cand in ranked_list:
                rank = ranked_list.index(correct_cand) + 1  # 1-based rank
                reciprocal_rank = 1.0 / rank
            else:
                reciprocal_rank = 0.0  # Not found = rank ∞
            
            total_reciprocal_rank += reciprocal_rank
            total_queries += 1


    mrr = total_reciprocal_rank / total_queries if total_queries > 0 else 0.0
    return mrr, total_queries

# Run
if __name__ == "__main__":
    ranked_file = sys.argv[1] if len(sys.argv) > 1 else "ranked_output.jsonl"
    mrr, num_queries = calculate_mrr(ranked_file)
    
    print(f"MRR@10: {mrr:.4f}")
    print(f"Queries processed: {num_queries}")
    print(f"Perfect score = 1.0000 (correct cand always #1)")
    print(f"Random baseline = 0.1000 (correct cand avg position 5)")
