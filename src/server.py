import flwr as fl
from flwr.server.history import History
from flwr.server.strategy import FedAvg
import tensorflow as tf
import numpy as np
import os
import sys
import json
from typing import Dict, List, Tuple
from flwr.common import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

class PrivacyAwareFedAvg(FedAvg):
    """Extended FedAvg with privacy tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_budget = []
        self.client_round_metrics = []
        self.server_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': []}

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        """Aggregate model updates and track privacy"""
        epsilons = [fit_res.metrics['epsilon'] for _, fit_res in results if 'epsilon' in fit_res.metrics]
        
        if epsilons:
            avg_epsilon = np.mean(epsilons)
            self.privacy_budget.append(avg_epsilon)
            print(f"\n[Round {server_round}] Avg Privacy Budget (epsilon) this round: {avg_epsilon:.2f}")
        
        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients"""
        if not results:
            return None, {}
        
        total_examples = sum(res.num_examples for _, res in results)
        accuracies = [res.num_examples * res.metrics.get("accuracy", 0.0) for _, res in results]
        
        if total_examples > 0:
            avg_accuracy = sum(accuracies) / total_examples
            self.client_round_metrics.append({'round': server_round, 'accuracy': avg_accuracy})
            print(f"[Round {server_round}] Aggregated Client-Side Accuracy: {avg_accuracy:.4f}")

        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        """Trigger a server-side evaluation"""
        loss, metrics = super().evaluate(server_round, parameters)
        for key, value in metrics.items():
            if key in self.server_history:
                self.server_history[key].append(value)
        return loss, metrics

def get_evaluate_fn(X_test, y_test):
    """Return evaluation function for server-side evaluation"""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_test.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.set_weights(parameters)
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
        
        results = model.evaluate(X_test, y_test, verbose=0)
        loss, accuracy, precision, recall, auc = results
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n{'='*60}\n[Round {server_round}] Global Model Evaluation\nLoss: {loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1_score:.4f}\n{'='*60}")
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "auc": auc}
    return evaluate

def save_results(history: History, strategy: PrivacyAwareFedAvg, noise_level: float):
    """Saves the results to a unique JSON file."""
    os.makedirs('results', exist_ok=True)
    
    server_metrics_processed = {key: [val for _, val in values] for key, values in history.metrics_centralized.items()}
    server_loss_processed = [val for _, val in history.losses_centralized]

    results_data = {
        'noise_multiplier': noise_level,
        'privacy_budget_per_round': strategy.privacy_budget,
        'final_server_metrics': {
            'loss': server_loss_processed,
            **server_metrics_processed
        }
    }
    
    filename = f'results/results_noise_{noise_level}.json'
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\n[OK] Results saved to {filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python server.py <noise_multiplier>")
        sys.exit(1)
    
    noise_val = float(sys.argv[1])
    
    print("="*60, f"DP-FedAvg Server (Noise: {noise_val})", "="*60, sep="\n")
    
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Initialize global model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_test.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())
    
    strategy = PrivacyAwareFedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(X_test, y_test),
        on_fit_config_fn=lambda rnd: {"epochs": 2, "batch_size": 64},
        initial_parameters=initial_parameters,
    )
    
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    save_results(history, strategy, noise_val)

if __name__ == "__main__":
    main()