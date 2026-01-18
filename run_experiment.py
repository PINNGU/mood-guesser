import time
import requests
import argparse
import json
import os

from src.utils.dataset_loader import load_data
from src.platform.client import PlatformClient


def wait_for_coordinators(urls, timeout=60):
    """Poll health endpoints until all coordinators are ready."""
    print("Waiting for coordinators to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        all_ready = True
        for url in urls:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code != 200:
                    all_ready = False
                    break
            except requests.RequestException:
                all_ready = False
                break
        
        if all_ready:
            print("All coordinators are ready!")
            return True
        
        time.sleep(2)
    
    print("Timeout waiting for coordinators")
    return False


def get_crdt_summary(coordinator_url):
    """Fetch CRDT state summary from coordinator."""
    try:
        response = requests.get(f"{coordinator_url}/crdt/summary", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException as e:
        print(f"Failed to get CRDT summary from {coordinator_url}: {e}")
        return None


def get_latest_model(coordinator_url):
    """Fetch latest global model from coordinator."""
    try:
        response = requests.get(f"{coordinator_url}/model/latest", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException as e:
        print(f"Failed to get latest model from {coordinator_url}: {e}")
        return None


def run_experiment(num_rounds, local_epochs, coordinator_urls):
    """Run the federated learning experiment."""
    
    # Load and split dataset
    print("Loading dataset...")
    platform_data, label_encoder, feature_cols, scaler = load_data(n_platforms=5)
    input_dim = len(feature_cols)
    output_dim = len(label_encoder.classes_)
    print(f"Dataset loaded: {input_dim} features, {output_dim} classes")
    
    # Create platform clients
    print("Creating platform clients...")
    platforms = []
    for i, data in enumerate(platform_data):
        platform = PlatformClient(
            platform_id=i,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            coordinator_urls=coordinator_urls,
            input_dim=input_dim,
            output_dim=output_dim
        )
        platforms.append(platform)
        print(f"  Platform {i}: {len(data['X_train'])} train samples, {len(data['X_test'])} test samples")
    
    # Wait for coordinators
    if not wait_for_coordinators(coordinator_urls):
        print("ERROR: Coordinators not available. Please start them first.")
        return None
    
    # Training results storage
    results = {
        'rounds': [],
        'final_accuracies': [],
        'average_accuracies': []
    }
    
    # Run training rounds
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Experiment")
    print(f"Rounds: {num_rounds}, Local Epochs per Round: {local_epochs}")
    print(f"{'='*60}\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"=== Round {round_num}/{num_rounds} ===")
        print(f"{'='*60}")
        
        round_results = {
            'round': round_num,
            'platforms': []
        }
        
        # Each platform trains locally and sends update
        for platform in platforms:
            # Train locally
            avg_loss = platform.train_local_epoch(epochs=local_epochs)
            
            # Evaluate
            accuracy = platform.evaluate()
            
            # Send update to coordinator
            platform.send_update_to_coordinator(platform.get_model_update())
            
            print(f"  Platform {platform.platform_id}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            round_results['platforms'].append({
                'platform_id': platform.platform_id,
                'loss': avg_loss,
                'accuracy': accuracy
            })
        
        # Wait a moment for aggregation to complete
        time.sleep(2)
        
        # Each platform fetches and loads the latest global model
        print("\nFetching global model...")
        for platform in platforms:
            model_response = get_latest_model(platform.get_coordinator_url())
            if model_response and 'model_state' in model_response:
                platform.receive_global_model(model_response['model_state'])
                print(f"  Platform {platform.platform_id}: Updated to model version {model_response.get('model_version', 'N/A')}")
            else:
                print(f"  Platform {platform.platform_id}: Failed to fetch global model")
        
        # Calculate average accuracy
        accuracies = [p['accuracy'] for p in round_results['platforms']]
        avg_accuracy = sum(accuracies) / len(accuracies)
        round_results['average_accuracy'] = avg_accuracy
        
        print(f"\n>>> Average Accuracy: {avg_accuracy:.2f}% <<<")
        
        results['rounds'].append(round_results)
        results['average_accuracies'].append(avg_accuracy)
        
        # Print CRDT summary every 5 rounds
        if round_num % 5 == 0:
            print(f"\n--- CRDT Summary (Round {round_num}) ---")
            for url in coordinator_urls:
                summary = get_crdt_summary(url)
                if summary:
                    print(f"  {url}: {json.dumps(summary, indent=2)}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("=== Final Results ===")
    print(f"{'='*60}")
    
    final_accuracies = []
    for platform in platforms:
        accuracy = platform.evaluate()
        final_accuracies.append(accuracy)
        print(f"  Platform {platform.platform_id}: Final Accuracy = {accuracy:.2f}%")
    
    results['final_accuracies'] = final_accuracies
    final_avg = sum(final_accuracies) / len(final_accuracies)
    print(f"\n>>> Final Average Accuracy: {final_avg:.2f}% <<<")
    
    # Print CRDT final summary
    print(f"\n--- Final CRDT Summary ---")
    for url in coordinator_urls:
        summary = get_crdt_summary(url)
        if summary:
            print(f"  {url}:")
            print(f"    {json.dumps(summary, indent=4)}")
    
    return results


def save_results(results, output_path):
    """Save experiment results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Federated Learning Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Round-by-Round Summary:\n")
        f.write("-" * 40 + "\n")
        for round_data in results['rounds']:
            f.write(f"\nRound {round_data['round']}:\n")
            for p in round_data['platforms']:
                f.write(f"  Platform {p['platform_id']}: Loss={p['loss']:.4f}, Accuracy={p['accuracy']:.2f}%\n")
            f.write(f"  Average Accuracy: {round_data['average_accuracy']:.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Final Results:\n")
        f.write("-" * 40 + "\n")
        for i, acc in enumerate(results['final_accuracies']):
            f.write(f"  Platform {i}: {acc:.2f}%\n")
        
        final_avg = sum(results['final_accuracies']) / len(results['final_accuracies'])
        f.write(f"\nFinal Average Accuracy: {final_avg:.2f}%\n")
        
        # Also save as JSON
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as jf:
            json.dump(results, jf, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"JSON results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment")
    parser.add_argument("--rounds", type=int, default=20, help="Number of training rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per round")
    
    args = parser.parse_args()
    
    # Coordinator URLs
    coordinator_urls = [
        'http://localhost:8001',
        'http://localhost:8002',
        'http://localhost:8003'
    ]
    
    print("Federated Learning Experiment")
    print(f"Coordinators: {coordinator_urls}")
    print(f"Rounds: {args.rounds}")
    print(f"Local Epochs: {args.local_epochs}")
    print()
    
    # Run experiment
    results = run_experiment(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        coordinator_urls=coordinator_urls
    )
    
    if results:
        # Save results
        save_results(results, 'results/experiment_results.txt')


if __name__ == "__main__":
    main()
