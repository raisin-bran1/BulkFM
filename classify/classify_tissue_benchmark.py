import subprocess
import re
import numpy as np
import pandas as pd

def run_classifier(embeddings, model_type, seed):
    cmd = [
        "python", "classify/classifier.py",
        "--embeddings", embeddings,
        "--model_type", model_type,
        "--seed", str(seed),
        "--column", "organ",
        "--epochs", "50",
        "--lr", "1e-4",
        "--batch_size", "64"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract accuracy using regex
        match = re.search(r"Accuracy: (\d+\.\d+)%", result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error running seed {seed} for {embeddings}/{model_type}: {e}")
    return None

def main():
    seeds = [1, 2, 3, 4, 5]
    configs = [
        ("osdr/osdr_embeddings.pt", "mlp"),
        ("osdr/osdr_embeddings.pt", "logistic"),
        ("osdr/osdr_embeddings_harmony.pt", "logistic")
    ]
    
    results = []
    
    for emb, model in configs:
        emb_name = "Binformer" if "osdr_embeddings.pt" in emb else "Harmony"
        print(f"\nBenchmarking {emb_name} + {model}...")
        accs = []
        for seed in seeds:
            acc = run_classifier(emb, model, seed)
            if acc is not None:
                accs.append(acc)
                print(f"  Seed {seed}: {acc}%")
        
        if accs:
            results.append({
                "Embeddings": emb_name,
                "Model": model,
                "Mean": np.mean(accs),
                "Std": np.std(accs),
                "Min": np.min(accs),
                "Max": np.max(accs)
            })

    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS (Across 5 Seeds)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
