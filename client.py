"""
DP-FedAvg Client Implementation (Manual DP-SGD Version)
------------------------------------------------------
This version implements Gradient Clipping and Noise Addition manually 
using TensorFlow GradientTape to avoid library version conflicts.
"""

import flwr as fl
import tensorflow as tf
import numpy as np
import sys
import warnings
import time

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')

class DPIoTClient(fl.client.NumPyClient):
    """IoT Client with Manual Differential Privacy implementation"""
    
    def __init__(self, client_id, X_train, y_train, noise_multiplier):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = 1.0  # Threshold for gradient clipping
        self.model = self.create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        print(f"[Client {client_id}] Initialized (Manual DP). Samples: {len(X_train)} | Noise: {noise_multiplier}")
    
    def create_model(self):
        """Create neural network model for anomaly detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def get_parameters(self, config):
        """Return current model parameters"""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """Train model with Manual DP-SGD (Clip + Noise)"""
        start_time = time.time()
        self.model.set_weights(parameters)
        
        epochs = config.get("epochs", 2)
        batch_size = config.get("batch_size", 64)
        
        # Prepare dataset for custom training loop
        ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        ds = ds.shuffle(len(self.X_train)).batch(batch_size)

        for epoch in range(epochs):
            for x_batch, y_batch in ds:
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    # Compute Binary Crossentropy Loss
                    loss = tf.keras.losses.binary_crossentropy(y_batch, logits)
                    loss = tf.reduce_mean(loss)
                
                # 1. Compute Raw Gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                if self.noise_multiplier > 0:
                    # 2. Gradient Clipping (Global Norm)
                    # Limits the influence of any single batch
                    gradients, _ = tf.clip_by_global_norm(gradients, self.l2_norm_clip)
                    
                    # 3. Add Gaussian Noise
                    # Noise std_dev is proportional to the sensitivity (clip) and noise multiplier
                    noisy_gradients = []
                    for grad in gradients:
                        noise = tf.random.normal(
                            shape=grad.shape, 
                            mean=0.0, 
                            stddev=self.noise_multiplier * self.l2_norm_clip
                        )
                        noisy_gradients.append(grad + noise)
                    gradients = noisy_gradients

                # 4. Apply the (noisy) gradients to the model
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Mathematical approximation of Privacy Budget (Epsilon)
        # Epsilon is inversely proportional to noise. 
        # This formula is a simplified approximation for the project report/graphs.
        if self.noise_multiplier > 0:
            epsilon = (np.sqrt(epochs * (len(self.X_train)/batch_size)) / self.noise_multiplier) * 2.0
        else:
            epsilon = 999.0 # Infinity = No privacy
        
        duration = time.time() - start_time
        
        # Evaluation on local data to report progress
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        _, acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        
        print(f"[Client {self.client_id}] Done | Accuracy: {acc:.4f} | Epsilon: {epsilon:.2f}")
        
        return self.model.get_weights(), len(self.X_train), {
            "accuracy": float(acc),
            "epsilon": float(epsilon)
        }
    
    def evaluate(self, parameters, config):
        """Evaluate model on local data"""
        self.model.set_weights(parameters)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"accuracy": accuracy}

def load_client_data(client_id):
    """Load and format data for a specific client"""
    X = np.load(f'data/clients/client_{client_id}_X.npy').astype(np.float32)
    y = np.load(f'data/clients/client_{client_id}_y.npy').astype(np.float32).reshape(-1, 1)
    return X, y

def start_client(client_id, noise_multiplier, server_address="127.0.0.1:8080"):
    """Initialize and start the Flower client"""
    X_train, y_train = load_client_data(client_id)
    client = DPIoTClient(client_id, X_train, y_train, noise_multiplier)
    
    # Using the newer Flower Client API structure
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == "__main__":
    import os
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <noise_multiplier>")
        sys.exit(1)
    
    cid = int(sys.argv[1])
    noise = float(sys.argv[2])
    
    # NEW: Check if we are running in Docker, otherwise use localhost
    addr = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    
    X_train, y_train = load_client_data(cid)
    client = DPIoTClient(cid, X_train, y_train, noise)
    
    fl.client.start_numpy_client(server_address=addr, client=client)