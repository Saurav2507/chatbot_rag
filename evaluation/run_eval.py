import json
import time
import requests

API_URL = "http://localhost:8000/api"

# Sample Ground Truth for Evaluation
GROUND_TRUTH = [
    {
        "query": "What is the primary function of the ingestion pipeline?",
        "expected_docs": ["architecture_doc.pdf"], # Can match on filename or substring of chunk
        "expected_keywords": ["extract", "chunk", "embed"]
    }
]

def evaluate():
    print("Starting evaluation...")
    results = []
    
    total_latency = 0
    correct_retrieval = 0
    
    for item in GROUND_TRUTH:
        start = time.time()
        try:
            response = requests.post(f"{API_URL}/chat", json={"query": item["query"], "top_k": 5})
            if response.status_code == 200:
                data = response.json()
                latency = time.time() - start
                total_latency += latency
                
                citations = data["citations"]
                retrieved_files = [c["filename"] for c in citations]
                
                # Check Recall@5 (did we get the right document?)
                hit = any(expected in retrieved for expected in item["expected_docs"] for retrieved in retrieved_files)
                if hit:
                    correct_retrieval += 1
                    
                results.append({
                    "query": item["query"],
                    "latency": latency,
                    "retrieved_files": retrieved_files,
                    "hit": hit
                })
            else:
                print(f"Error querying API for {item['query']}")
        except Exception as e:
            print(f"Connection failed: {e}")
            return
            
    avg_latency = total_latency / len(GROUND_TRUTH) if GROUND_TRUTH else 0
    recall_at_5 = (correct_retrieval / len(GROUND_TRUTH)) * 100 if GROUND_TRUTH else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Average Latency (e2e): {avg_latency:.2f}s")
    print(f"Recall@5: {recall_at_5:.1f}%")
    
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate()
