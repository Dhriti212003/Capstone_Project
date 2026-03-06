"""
Automated Experiment Runner for DP-FedAvg
Runs server and clients in separate processes (Windows Compatible)
"""

import subprocess
import time
import sys
import os
import signal

def start_server():
    """Start the federated learning server"""
    print("\n[ORCHESTRATOR] Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return server_process

def start_client(client_id):
    """Start a federated learning client"""
    print(f"[ORCHESTRATOR] Starting client {client_id}...")
    client_process = subprocess.Popen(
        [sys.executable, "client.py", str(client_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return client_process

def stream_output(process, prefix):
    """Stream process output in real-time"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{prefix}] {line.rstrip()}")
        process.stdout.close()
    except Exception as e:
        print(f"[{prefix}] Error reading output: {e}")

def run_experiment(num_clients=3):
    """Run complete federated learning experiment"""
    
    print("="*70)
    print(" Privacy-Preserving Threat Detection using DP-FedAvg")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Number of clients: {num_clients}")
    print(f"  - Privacy: Differential Privacy enabled")
    print(f"  - Algorithm: Federated Averaging (FedAvg)")
    print("="*70)
    
    processes = []
    
    try:
        # Start server
        server_proc = start_server()
        processes.append(('Server', server_proc))
        
        # Wait for server to initialize
        print("\n[ORCHESTRATOR] Waiting for server initialization...")
        time.sleep(5)
        
        # Start clients
        client_procs = []
        for i in range(num_clients):
            client_proc = start_client(i)
            client_procs.append(client_proc)
            processes.append((f'Client-{i}', client_proc))
            time.sleep(2)  # Stagger client starts
        
        print("\n[ORCHESTRATOR] All processes started. Training in progress...")
        print("[ORCHESTRATOR] Press Ctrl+C to stop\n")
        
        # Monitor processes
        import threading
        
        # Stream server output
        server_thread = threading.Thread(
            target=stream_output, 
            args=(server_proc, "SERVER")
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Stream client outputs
        client_threads = []
        for i, client_proc in enumerate(client_procs):
            thread = threading.Thread(
                target=stream_output,
                args=(client_proc, f"CLIENT-{i}")
            )
            thread.daemon = True
            thread.start()
            client_threads.append(thread)
        
        # Wait for server to complete
        server_proc.wait()
        
        print("\n[ORCHESTRATOR] Server finished. Waiting for clients to terminate...")
        time.sleep(2)
        
        # Terminate clients if still running
        for name, proc in processes[1:]:  # Skip server
            if proc.poll() is None:
                proc.terminate()
                print(f"[ORCHESTRATOR] Terminated {name}")
        
        print("\n" + "="*70)
        print("Experiment Complete!")
        print("="*70)
        print("\nResults saved in:")
        print("  - results/training_results.json")
        print("\nTo visualize results, run: python visualize_results.py")
        
    except KeyboardInterrupt:
        print("\n\n[ORCHESTRATOR] Interrupted by user. Cleaning up...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"[ORCHESTRATOR] Terminated {name}")
        print("[ORCHESTRATOR] Cleanup complete")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ORCHESTRATOR] Error: {e}")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    # Check if data is prepared
    if not os.path.exists('data/clients'):
        print("ERROR: Dataset not prepared!")
        print("Please run: python prepare_dataset.py")
        sys.exit(1)
    
    run_experiment(num_clients=3)