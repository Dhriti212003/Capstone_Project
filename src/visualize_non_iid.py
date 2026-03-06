import numpy as np
import matplotlib.pyplot as plt
import os

def plot_non_iid_distribution():
    """
    Reads the prepared client data and creates a bar chart showing 
    the class imbalance (Normal vs Attack) for each simulated IoT device.
    """
    client_counts = []
    
    # We check the 3 clients we created in prepare_dataset.py
    for i in range(3):
        path = f'data/clients/client_{i}_y.npy'
        if not os.path.exists(path):
            print(f"Error: {path} not found. Please run prepare_dataset.py first.")
            return
            
        y = np.load(path)
        normal = np.sum(y == 0)
        attack = np.sum(y == 1)
        client_counts.append([normal, attack])
    
    client_counts = np.array(client_counts)
    
    # These labels match the logic in your prepare_dataset.py
    labels = ['Client 0\n(Mostly Normal)', 'Client 1\n(Highly Targeted)', 'Client 2\n(Balanced Gateway)']
    normal_traffic = client_counts[:, 0]
    attack_traffic = client_counts[:, 1]

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use('ggplot') # Makes the chart look professional
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, normal_traffic, width, label='Normal Traffic', color='#2ecc71')
    rects2 = ax.bar(x + width/2, attack_traffic, width, label='Attack/Threat', color='#e74c3c')

    ax.set_ylabel('Number of Log Entries')
    ax.set_title('Non-IID Data Distribution Across IoT Edge Nodes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add count labels on top of bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/non_iid_distribution.png', dpi=300)
    print("\n[OK] Non-IID distribution chart saved to: results/non_iid_distribution.png")
    plt.show()

if __name__ == "__main__":
    plot_non_iid_distribution()