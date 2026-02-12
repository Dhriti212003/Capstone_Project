import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_all_results():
    results = {}
    noise_levels = [0.0, 0.5, 1.1, 2.0, 4.0]
    
    for noise in noise_levels:
        filename = f'results/results_noise_{noise}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                results[noise] = json.load(f)
        else:
            print(f"Warning: {filename} not found. Skipping.")
    return results

def plot_tradeoff_analysis(all_data):
    noise_levels = sorted(all_data.keys())
    final_accuracies = []
    epsilons = []

    for noise in noise_levels:
        # Get the last accuracy value from the 10 rounds
        acc = all_data[noise]['final_server_metrics']['accuracy'][-1]
        final_accuracies.append(acc)
        
        # Get the last epsilon value (privacy budget)
        eps = all_data[noise]['privacy_budget_per_round'][-1]
        epsilons.append(eps)

    # --- CHART 1: ACCURACY VS NOISE ---
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, final_accuracies, marker='o', linewidth=2, color='#3498db')
    plt.title('Impact of Noise Multiplier on Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Multiplier (Higher = More Privacy)', fontsize=12)
    plt.ylabel('Final Global Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    for i, txt in enumerate(final_accuracies):
        plt.annotate(f"{txt:.2%}", (noise_levels[i], final_accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig('results/accuracy_vs_noise.png', dpi=300)
    print("[OK] Saved: results/accuracy_vs_noise.png")

    # --- CHART 2: THE PRIVACY-UTILITY TRADE-OFF (EPSILON VS ACCURACY) ---
    plt.figure(figsize=(10, 6))
    # Filter out noise 0.0 (epsilon 999) for a cleaner graph
    clean_eps = [e for e in epsilons if e < 100]
    clean_acc = [final_accuracies[i] for i, e in enumerate(epsilons) if e < 100]
    
    plt.scatter(clean_eps, clean_acc, s=100, color='#e74c3c', edgecolors='black')
    plt.plot(clean_eps, clean_acc, linestyle='--', alpha=0.5, color='#e74c3c')
    plt.title('Privacy-Utility Trade-off (Epsilon vs Accuracy)', fontsize=14, fontweight='bold')
    plt.xlabel('Privacy Budget (Epsilon) - LOWER is more private', fontsize=12)
    plt.ylabel('Global Accuracy', fontsize=12)
    plt.gca().invert_xaxis() # Lower epsilon is stronger privacy
    plt.grid(True, alpha=0.3)
    plt.savefig('results/epsilon_vs_accuracy.png', dpi=300)
    print("[OK] Saved: results/epsilon_vs_accuracy.png")

    # --- CHART 3: CONVERGENCE PER NOISE LEVEL ---
    plt.figure(figsize=(10, 6))
    for noise in noise_levels:
        acc_history = all_data[noise]['final_server_metrics']['accuracy']
        label = f"Noise {noise} (Baseline)" if noise == 0.0 else f"Noise {noise}"
        plt.plot(range(len(acc_history)), acc_history, label=label, linewidth=2)
    
    plt.title('Global Model Convergence Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/convergence_comparison.png', dpi=300)
    print("[OK] Saved: results/convergence_comparison.png")

def generate_final_table(all_data):
    print("\n" + "="*50)
    print(f"{'Noise':<10} | {'Epsilon':<10} | {'Final Accuracy':<15}")
    print("-" * 50)
    for noise in sorted(all_data.keys()):
        acc = all_data[noise]['final_server_metrics']['accuracy'][-1]
        eps = all_data[noise]['privacy_budget_per_round'][-1]
        eps_str = "Inf (No DP)" if eps > 100 else f"{eps:.2f}"
        print(f"{noise:<10} | {eps_str:<10} | {acc:<15.4f}")
    print("="*50)

if __name__ == "__main__":
    data = load_all_results()
    if data:
        plot_tradeoff_analysis(data)
        generate_final_table(data)
    else:
        print("No result files found in /results folder.")