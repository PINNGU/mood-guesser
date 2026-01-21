import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def calculate_confusion_matrix(y_true, y_pred, label_encoder):
    """
    Calculate confusion matrix for predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: LabelEncoder used for encoding labels
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_training_progress(round_history, output_path='results/training_progress.png'):
    """
    Plot training progress over federated learning rounds.
    
    Args:
        round_history: List of dicts with {'round': X, 'accuracy': Y}
        output_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rounds = [entry['round'] for entry in round_history]
    accuracies = [entry['accuracy'] for entry in round_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Federated Learning Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Training progress plot saved to {output_path}")


def compare_centralized_vs_federated(centralized_acc, federated_acc, output_path='results/comparison.png'):
    """
    Create bar chart comparing centralized vs federated learning accuracy.
    
    Args:
        centralized_acc: Accuracy of centralized model
        federated_acc: Accuracy of federated model
        output_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    methods = ['Centralized', 'Federated']
    accuracies = [centralized_acc, federated_acc]
    colors = ['#2ecc71', '#3498db']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Learning Method', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Centralized vs Federated Learning Comparison', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")


def generate_report(final_results, output_path='results/report.txt'):
    """
    Generate a formatted text report of the experiment results.
    
    Args:
        final_results: Dict with accuracy, confusion_matrix, and other metrics
        output_path: Path to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("       MOOD CLASSIFICATION EXPERIMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Accuracy metrics
        f.write("-" * 40 + "\n")
        f.write("ACCURACY METRICS\n")
        f.write("-" * 40 + "\n")
        
        if 'accuracy' in final_results:
            f.write(f"Overall Accuracy: {final_results['accuracy']:.4f}\n")
        
        if 'centralized_accuracy' in final_results:
            f.write(f"Centralized Accuracy: {final_results['centralized_accuracy']:.4f}\n")
        
        if 'federated_accuracy' in final_results:
            f.write(f"Federated Accuracy: {final_results['federated_accuracy']:.4f}\n")
        
        f.write("\n")
        
        # Training statistics
        if 'num_rounds' in final_results:
            f.write("-" * 40 + "\n")
            f.write("TRAINING STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of Rounds: {final_results['num_rounds']}\n")
            
            if 'num_clients' in final_results:
                f.write(f"Number of Clients: {final_results['num_clients']}\n")
            
            if 'samples_per_client' in final_results:
                f.write(f"Samples per Client: {final_results['samples_per_client']}\n")
            
            f.write("\n")
        
        # Confusion matrix
        if 'confusion_matrix' in final_results:
            f.write("-" * 40 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            cm = final_results['confusion_matrix']
            if isinstance(cm, np.ndarray):
                f.write(np.array2string(cm, separator=', '))
            else:
                f.write(str(cm))
            f.write("\n\n")
        
        # Class labels
        if 'class_labels' in final_results:
            f.write("-" * 40 + "\n")
            f.write("CLASS LABELS\n")
            f.write("-" * 40 + "\n")
            for i, label in enumerate(final_results['class_labels']):
                f.write(f"  {i}: {label}\n")
            f.write("\n")
        
        # Per-class metrics
        if 'per_class_accuracy' in final_results:
            f.write("-" * 40 + "\n")
            f.write("PER-CLASS ACCURACY\n")
            f.write("-" * 40 + "\n")
            for label, acc in final_results['per_class_accuracy'].items():
                f.write(f"  {label}: {acc:.4f}\n")
            f.write("\n")
        
        # Additional notes
        if 'notes' in final_results:
            f.write("-" * 40 + "\n")
            f.write("NOTES\n")
            f.write("-" * 40 + "\n")
            f.write(final_results['notes'] + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("                 END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"Report saved to {output_path}")
