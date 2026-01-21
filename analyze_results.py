import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RESULTS_FILE = "results/experiment_results.json"
RESULTS_DIR = "results"


def load_results(filepath: str = RESULTS_FILE) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {filepath}")
    return results


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")


def plot_accuracy_over_rounds(results: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(12, 6))
    
    platform_results = results.get('platform_results', {})
    
    for platform_id, data in platform_results.items():
        accuracies = data.get('accuracy_history', [])
        rounds = list(range(1, len(accuracies) + 1))
        plt.plot(rounds, accuracies, marker='o', markersize=4, 
                linewidth=2, label=f'Platform {platform_id}')
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Over Training Rounds (All Platforms)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_average_accuracy(results: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(12, 6))
    
    platform_results = results.get('platform_results', {})
    
    all_accuracies = []
    max_rounds = 0
    for data in platform_results.values():
        acc_history = data.get('accuracy_history', [])
        all_accuracies.append(acc_history)
        max_rounds = max(max_rounds, len(acc_history))
    
    for i, acc in enumerate(all_accuracies):
        if len(acc) < max_rounds:
            all_accuracies[i] = acc + [acc[-1]] * (max_rounds - len(acc))
    
    acc_array = np.array(all_accuracies)
    mean_acc = np.mean(acc_array, axis=0)
    std_acc = np.std(acc_array, axis=0)
    rounds = list(range(1, max_rounds + 1))
    
    plt.plot(rounds, mean_acc, 'b-', linewidth=2, label='Mean Accuracy')
    plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Average Model Accuracy Over Training Rounds', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(results: Dict[str, Any], save_path: str = None):
    confusion_matrix = results.get('confusion_matrix', None)
    
    if confusion_matrix is None:
        print("No confusion matrix found in results, skipping...")
        return
    
    if isinstance(confusion_matrix, list):
        cm = np.array(confusion_matrix)
    else:
        cm = confusion_matrix
    
    labels = results.get('class_labels', [f'Class {i}' for i in range(cm.shape[0])])
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                square=True, linewidths=0.5)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Aggregated)', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_final_accuracies(results: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(10, 6))
    
    platform_results = results.get('platform_results', {})
    
    platforms = []
    final_accuracies = []
    
    for platform_id, data in sorted(platform_results.items()):
        platforms.append(f'Platform {platform_id}')
        acc_history = data.get('accuracy_history', [0])
        final_accuracies.append(acc_history[-1] if acc_history else 0)
    
    colors = ['#2ecc71' if acc >= 0.8 else '#f39c12' if acc >= 0.6 else '#e74c3c' 
              for acc in final_accuracies]
    
    bars = plt.bar(platforms, final_accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='80% Target')
    
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.title('Final Model Accuracy by Platform', fontsize=14)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_crdt_state_changes(results: Dict[str, Any], save_path: str = None):
    crdt_history = results.get('crdt_history', [])
    
    if not crdt_history:
        print("No CRDT history found in results, skipping...")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    rounds = [entry.get('round', i) for i, entry in enumerate(crdt_history)]
    training_rounds = [entry.get('training_rounds', 0) for entry in crdt_history]
    total_samples = [entry.get('total_samples', 0) for entry in crdt_history]
    
    ax1 = axes[0]
    ax1.plot(rounds, training_rounds, 'b-o', linewidth=2, markersize=6, label='Training Rounds')
    ax1.set_xlabel('Experiment Round', fontsize=11)
    ax1.set_ylabel('Training Rounds (cumulative)', fontsize=11, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rounds, total_samples, 'r-s', linewidth=2, markersize=6, label='Total Samples')
    ax1_twin.set_ylabel('Total Samples', fontsize=11, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('CRDT State: Training Progress Over Time', fontsize=13)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2 = axes[1]
    
    mood_data = {}
    for entry in crdt_history:
        mood_counts = entry.get('mood_counts', {})
        for mood, count in mood_counts.items():
            if mood not in mood_data:
                mood_data[mood] = []
            mood_data[mood].append(count)
    
    max_len = len(rounds)
    for mood in mood_data:
        while len(mood_data[mood]) < max_len:
            mood_data[mood].append(mood_data[mood][-1] if mood_data[mood] else 0)
    
    for mood, counts in mood_data.items():
        ax2.plot(rounds[:len(counts)], counts, '-o', linewidth=2, markersize=5, label=mood.capitalize())
    
    ax2.set_xlabel('Experiment Round', fontsize=11)
    ax2.set_ylabel('Mood Count', fontsize=11)
    ax2.set_title('CRDT State: Mood Counts Over Time', fontsize=13)
    ax2.legend(loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def calculate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    stats = {}
    platform_results = results.get('platform_results', {})
    
    final_accuracies = []
    for data in platform_results.values():
        acc_history = data.get('accuracy_history', [0])
        final_accuracies.append(acc_history[-1] if acc_history else 0)
    
    stats['final_average_accuracy'] = np.mean(final_accuracies)
    stats['accuracy_std_dev'] = np.std(final_accuracies)
    stats['min_accuracy'] = np.min(final_accuracies)
    stats['max_accuracy'] = np.max(final_accuracies)
    
    all_accuracies = []
    max_rounds = 0
    for data in platform_results.values():
        acc_history = data.get('accuracy_history', [])
        all_accuracies.append(acc_history)
        max_rounds = max(max_rounds, len(acc_history))
    
    convergence_round = None
    if all_accuracies:
        for i, acc in enumerate(all_accuracies):
            if len(acc) < max_rounds:
                all_accuracies[i] = acc + [acc[-1]] * (max_rounds - len(acc))
        
        acc_array = np.array(all_accuracies)
        mean_per_round = np.mean(acc_array, axis=0)
        
        for round_idx, avg_acc in enumerate(mean_per_round):
            if avg_acc >= 0.8:
                convergence_round = round_idx + 1
                break
    
    stats['convergence_round'] = convergence_round
    stats['total_rounds'] = max_rounds
    
    stats['total_training_time'] = results.get('total_time', 0)
    stats['num_platforms'] = len(platform_results)
    
    model_size_kb = results.get('model_size_kb', 0)
    stats['model_size_kb'] = model_size_kb
    stats['communication_cost_kb'] = model_size_kb * max_rounds * len(platform_results)
    
    crdt_history = results.get('crdt_history', [])
    if crdt_history:
        final_crdt = crdt_history[-1]
        stats['final_training_rounds'] = final_crdt.get('training_rounds', 0)
        stats['final_total_samples'] = final_crdt.get('total_samples', 0)
        stats['final_mood_counts'] = final_crdt.get('mood_counts', {})
    
    return stats


def print_statistical_summary(stats: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("           EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n--- Accuracy Metrics ---")
    print(f"  Final Average Accuracy:  {stats['final_average_accuracy']:.2%}")
    print(f"  Standard Deviation:      {stats['accuracy_std_dev']:.4f}")
    print(f"  Min Accuracy:            {stats['min_accuracy']:.2%}")
    print(f"  Max Accuracy:            {stats['max_accuracy']:.2%}")
    
    print("\n--- Training Progress ---")
    print(f"  Total Rounds:            {stats['total_rounds']}")
    convergence = stats['convergence_round']
    if convergence:
        print(f"  Convergence Round (80%): {convergence}")
    else:
        print(f"  Convergence Round (80%): Not reached")
    print(f"  Number of Platforms:     {stats['num_platforms']}")
    
    print("\n--- Time & Communication ---")
    total_time = stats['total_training_time']
    if total_time > 0:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"  Total Training Time:     {minutes}m {seconds:.1f}s")
    else:
        print(f"  Total Training Time:     N/A")
    
    model_size = stats.get('model_size_kb', 0)
    if model_size > 0:
        print(f"  Model Size:              {model_size:.2f} KB")
        print(f"  Total Communication:     {stats['communication_cost_kb']:.2f} KB")
    
    if 'final_training_rounds' in stats:
        print("\n--- CRDT Final State ---")
        print(f"  Training Rounds:         {stats['final_training_rounds']}")
        print(f"  Total Samples:           {stats['final_total_samples']}")
        
        mood_counts = stats.get('final_mood_counts', {})
        if mood_counts:
            print(f"  Mood Distribution:")
            total_moods = sum(mood_counts.values())
            for mood, count in sorted(mood_counts.items()):
                pct = (count / total_moods * 100) if total_moods > 0 else 0
                print(f"    - {mood}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)


def generate_all_plots(results: Dict[str, Any]):
    ensure_results_dir()
    
    print("\nGenerating visualizations...")
    
    plot_accuracy_over_rounds(
        results, 
        save_path=os.path.join(RESULTS_DIR, "accuracy_over_rounds.png")
    )
    
    plot_average_accuracy(
        results,
        save_path=os.path.join(RESULTS_DIR, "average_accuracy.png")
    )
    
    plot_confusion_matrix(
        results,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )
    
    plot_final_accuracies(
        results,
        save_path=os.path.join(RESULTS_DIR, "final_accuracies.png")
    )
    
    plot_crdt_state_changes(
        results,
        save_path=os.path.join(RESULTS_DIR, "crdt_state_changes.png")
    )
    
    print("\nAll visualizations saved to results/ folder")


def main():
    print("=" * 60)
    print("    Federated Learning Experiment Analysis")
    print("=" * 60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = load_results()
        
        generate_all_plots(results)
        
        stats = calculate_statistics(results)
        print_statistical_summary(stats)
        
        stats_file = os.path.join(RESULTS_DIR, "statistics.json")
        
        stats_serializable = {}
        for key, value in stats.items():
            if isinstance(value, np.floating):
                stats_serializable[key] = float(value)
            elif isinstance(value, np.integer):
                stats_serializable[key] = int(value)
            else:
                stats_serializable[key] = value
        
        with open(stats_file, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
        
        print("\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the experiment first using run_experiment.py")
        print("The experiment should save results to: results/experiment_results.json")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
