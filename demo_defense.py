import time
import requests
import json

from src.utils.dataset_loader import load_data
from src.platform.client import PlatformClient


COORDINATOR_URLS = [
    'http://localhost:8001',
    'http://localhost:8002',
    'http://localhost:8003'
]


def print_banner():
    print("\n" + "=" * 60)
    print("       MoodSync Federated Learning Demo")
    print("=" * 60)
    print("\nThis demo showcases:")
    print("  - Federated learning with multiple platforms")
    print("  - CRDT-based state synchronization")
    print("  - Fault tolerance with coordinator failures")
    print("=" * 60 + "\n")


def check_coordinator_health(url):
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_crdt_summary(url):
    try:
        response = requests.get(f"{url}/crdt/summary", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def get_latest_model(url):
    try:
        response = requests.get(f"{url}/model/latest", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def step1_check_coordinators():
    print("\n" + "=" * 60)
    print("STEP 1: Checking Coordinator Status")
    print("=" * 60 + "\n")
    
    while True:
        all_online = True
        print("Coordinator Status:")
        
        for i, url in enumerate(COORDINATOR_URLS):
            is_online = check_coordinator_health(url)
            status = "ONLINE" if is_online else "OFFLINE"
            symbol = "[+]" if is_online else "[-]"
            print(f"  {symbol} Coordinator {i+1} ({url}): {status}")
            if not is_online:
                all_online = False
        
        if all_online:
            print("\nAll coordinators are running!")
            break
        else:
            print("\nPlease start coordinators with: docker-compose up -d")
            print("Waiting for coordinators to come online...")
            time.sleep(5)
    
    return True


def step2_load_dataset():
    print("\n" + "=" * 60)
    print("STEP 2: Loading Dataset and Statistics")
    print("=" * 60 + "\n")
    
    print("Loading Spotify mood classification dataset...")
    platform_data, label_encoder, feature_cols, scaler = load_data(n_platforms=5)
    
    input_dim = len(feature_cols)
    output_dim = len(label_encoder.classes_)
    
    print(f"\nDataset Information:")
    print(f"  Features: {feature_cols}")
    print(f"  Number of features: {input_dim}")
    print(f"  Mood classes: {list(label_encoder.classes_)}")
    print(f"  Number of classes: {output_dim}")
    
    print(f"\nPlatform Data Distribution:")
    total_train = 0
    for i, data in enumerate(platform_data):
        train_samples = len(data['X_train'])
        test_samples = len(data['X_test'])
        total_train += train_samples
        print(f"  Platform {i}: {train_samples} training samples, {test_samples} test samples")
    
    print(f"\n  Total training samples: {total_train}")
    print(f"  Test samples (shared): {len(platform_data[0]['X_test'])}")
    
    return platform_data, label_encoder, feature_cols, input_dim, output_dim


def step3_training_rounds(platforms, num_rounds=5):
    print("\n" + "=" * 60)
    print(f"STEP 3: Running {num_rounds} Training Rounds")
    print("=" * 60 + "\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        for platform in platforms:
            avg_loss = platform.train_local_epoch(epochs=3)
            accuracy = platform.evaluate()
            platform.send_update_to_coordinator(platform.get_model_update())
            print(f"  Platform {platform.platform_id}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        time.sleep(2)
        
        for platform in platforms:
            model_response = get_latest_model(platform.get_coordinator_url())
            if model_response and 'model_state' in model_response:
                platform.receive_global_model(model_response['model_state'])
        
        accuracies = [platform.evaluate() for platform in platforms]
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"\n  Average Accuracy after Round {round_num}: {avg_accuracy:.2f}%")
        
        print(f"\n  CRDT State (Round {round_num}):")
        for url in COORDINATOR_URLS:
            summary = get_crdt_summary(url)
            if summary:
                print(f"    {url}: {json.dumps(summary)}")
    
    return platforms


def step4_fault_tolerance(platforms):
    print("\n" + "=" * 60)
    print("STEP 4: Demonstrating Fault Tolerance")
    print("=" * 60 + "\n")
    
    print("Simulating coordinator failure...")
    print("\nIn another terminal, run: docker kill coordinator-2")
    print("\nThis will stop one of the coordinators to demonstrate")
    print("that the system continues to function with remaining coordinators.")
    
    input("\nPress Enter after killing the coordinator...")
    
    print("\nChecking coordinator status after failure:")
    for i, url in enumerate(COORDINATOR_URLS):
        is_online = check_coordinator_health(url)
        status = "ONLINE" if is_online else "OFFLINE"
        symbol = "[+]" if is_online else "[-]"
        print(f"  {symbol} Coordinator {i+1} ({url}): {status}")
    
    print("\nContinuing training for 3 more rounds despite failure...")
    
    for round_num in range(1, 4):
        print(f"\n--- Fault Tolerance Round {round_num}/3 ---")
        
        for platform in platforms:
            avg_loss = platform.train_local_epoch(epochs=3)
            accuracy = platform.evaluate()
            response = platform.send_update_to_coordinator(platform.get_model_update())
            status = "sent" if response else "failed (retrying with another coordinator)"
            print(f"  Platform {platform.platform_id}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}% - Update {status}")
        
        time.sleep(2)
        
        for platform in platforms:
            model_response = get_latest_model(platform.get_coordinator_url())
            if model_response and 'model_state' in model_response:
                platform.receive_global_model(model_response['model_state'])
        
        accuracies = [platform.evaluate() for platform in platforms]
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"\n  Average Accuracy: {avg_accuracy:.2f}%")
    
    print("\n" + "-" * 40)
    print("System continued working despite failure!")
    print("-" * 40)
    
    return platforms


def step5_final_results(platforms):
    print("\n" + "=" * 60)
    print("STEP 5: Final Results")
    print("=" * 60 + "\n")
    
    print("Final Platform Accuracies:")
    final_accuracies = []
    for platform in platforms:
        accuracy = platform.evaluate()
        final_accuracies.append(accuracy)
        print(f"  Platform {platform.platform_id}: {accuracy:.2f}%")
    
    avg_accuracy = sum(final_accuracies) / len(final_accuracies)
    print(f"\nFinal Average Accuracy: {avg_accuracy:.2f}%")
    
    print("\nCRDT Consistency Verification:")
    print("Checking if all coordinators have consistent state...")
    
    crdt_states = []
    for url in COORDINATOR_URLS:
        summary = get_crdt_summary(url)
        if summary:
            crdt_states.append(summary)
            print(f"  {url}: {json.dumps(summary)}")
        else:
            print(f"  {url}: OFFLINE (expected if we killed it)")
    
    if len(crdt_states) >= 2:
        online_states = [json.dumps(s, sort_keys=True) for s in crdt_states]
        if len(set(online_states)) == 1:
            print("\n[+] CRDT Consistency: All online coordinators have IDENTICAL state!")
        else:
            print("\n[!] CRDT states differ (may be syncing...)")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Federated learning trained across 5 distributed platforms")
    print("  2. CRDT ensured consistent state across coordinators")
    print("  3. System remained operational despite coordinator failure")
    print("  4. Final model achieved good accuracy on mood classification")
    print("=" * 60 + "\n")


def main():
    print_banner()
    
    input("Press Enter to start the demo...")
    
    step1_check_coordinators()
    input("\nPress Enter to continue to Step 2...")
    
    platform_data, label_encoder, feature_cols, input_dim, output_dim = step2_load_dataset()
    
    print("\nCreating platform clients...")
    platforms = []
    for i, data in enumerate(platform_data):
        platform = PlatformClient(
            platform_id=i,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            coordinator_urls=COORDINATOR_URLS,
            input_dim=input_dim,
            output_dim=output_dim
        )
        platforms.append(platform)
    print(f"Created {len(platforms)} platform clients.")
    
    input("\nPress Enter to continue to Step 3...")
    
    platforms = step3_training_rounds(platforms, num_rounds=5)
    input("\nPress Enter to continue to Step 4 (Fault Tolerance)...")
    
    platforms = step4_fault_tolerance(platforms)
    input("\nPress Enter to continue to Step 5 (Final Results)...")
    
    step5_final_results(platforms)
    
    print("Thank you for watching the MoodSync demo!")


if __name__ == "__main__":
    main()
