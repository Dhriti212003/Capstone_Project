"""
Results Visualization for DP-FedAvg Project (Definitive Version)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results():
    """Loads the training results from the results.json file."""
    with open('results/training_results.json', 'r') as f:
        return json.load(f)

def plot_performance_over_rounds(metrics):
    """Plots the global model's accuracy and loss over the rounds."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    accuracy_values = metrics.get('accuracy', [])
    loss_values = metrics.get('loss', [])
    rounds = list(range(len(accuracy_values)))

    if not accuracy_values or not loss_values:
        print("ERROR: Could not find 'accuracy' or 'loss' data in results file.")
        return

    # Plot Accuracy
    color = 'tab:green'
    ax1.set_xlabel('Federated Round (0=initial)', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color, fontsize=12)
    ax1.plot(rounds, accuracy_values, 'o-', color=color, label='Global Model Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([min(accuracy_values) * 0.95, 1.0])

    # Plot Loss on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color, fontsize=12)
    ax2.plot(rounds, loss_values, 's--', color=color, label='Global Model Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Global Model Performance on Test Set', fontsize=16)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.savefig('results/performance_over_rounds.png', dpi=300)
    print("-> [OK] Saved: results/performance_over_rounds.png")
    plt.close()

def plot_privacy_accuracy_tradeoff(results):
    """Plots the tradeoff between cumulative privacy cost and model accuracy."""
    privacy_budget = results.get('privacy_budget_per_round', [])
    accuracy_values = results.get('final_server_metrics', {}).get('accuracy', [])

    if not privacy_budget or not accuracy_values:
        return

    # Align data: privacy budget starts at round 1, so we skip the initial accuracy
    accuracy_values_after_round_0 = accuracy_values[1:]
    cumulative_epsilon = np.cumsum(privacy_budget)
    
    min_len = min(len(cumulative_epsilon), len(accuracy_values_after_round_0))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.scatter(cumulative_epsilon[:min_len], accuracy_values_after_round_0[:min_len], 
                c=range(1, min_len + 1), cmap='viridis', s=120, edgecolors='k')
    
    cbar = plt.colorbar()
    cbar.set_label('Federated Round', fontsize=12)
    plt.xlabel('Cumulative Privacy Budget (epsilon)', fontsize=12)
    plt.ylabel('Global Model Accuracy', fontsize=12)
    plt.title('Privacy-Accuracy Tradeoff', fontsize=16)
    plt.grid(True)
    plt.savefig('results/privacy_accuracy_tradeoff.png', dpi=300)
    print("-> [OK] Saved: results/privacy_accuracy_tradeoff.png")
    plt.close()
    
def plot_f1_score(metrics):
    """Plots F1-Score over rounds to show attack detection reliability."""
    if 'f1_score' not in metrics:
        return
        
    plt.figure(figsize=(10, 5))
    f1_values = metrics['f1_score']
    rounds = range(len(f1_values))
    
    plt.plot(rounds, f1_values, 'o-', color='tab:orange', linewidth=2, label='Global F1-Score')
    plt.title('Attack Detection Reliability (F1-Score)', fontsize=14)
    plt.xlabel('Communication Round')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/f1_score_trend.png', dpi=300)
    print("-> [OK] Saved: results/f1_score_trend.png")

def generate_summary_report(results):
    """Generates a text summary of the final results."""
    report = ["="*70, "DP-FedAvg Training Summary Report", "="*70, ""]
    
    privacy_budget = results.get('privacy_budget_per_round', [])
    metrics = results.get('final_server_metrics', {})

    if privacy_budget:
        total_epsilon = sum(privacy_budget)
        report.extend([f"Total Privacy Budget (epsilon) Spent: {total_epsilon:.2f}", ""])

    if metrics.get('accuracy') and metrics.get('f1_score'):
        final_accuracy = metrics['accuracy'][-1]
        final_f1 = metrics['f1_score'][-1]
        report.extend([
            "--- Final Global Model Performance ---",
            f"Final Accuracy: {final_accuracy:.4f}",
            f"Final F1-Score: {final_f1:.4f}",
            ""
        ])
    
    report.append("="*70)
    print("\n".join(report))
    
    with open('results/summary_report.txt', 'w') as f:
        f.write("\n".join(report))
    print("\n-> [OK] Summary report saved to results/summary_report.txt")

def main():
    """Main function to load results and generate all outputs."""
    print("="*70, "Generating Visualizations and Reports", "="*70, sep="\n")
    
    results_path = 'results/training_results.json'
    if not os.path.exists(results_path):
        print("\nERROR: Please run 'python run_experiment.py' first to generate results.")
        return
        
    results = load_results()
    
    plot_performance_over_rounds(results.get('final_server_metrics', {}))
    plot_privacy_accuracy_tradeoff(results)
    generate_summary_report(results)

if __name__ == "__main__":
    main()