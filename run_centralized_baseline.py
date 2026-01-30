import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import numpy as np

from src.utils.dataset_loader import load_data, MoodDataset
from src.models.mood_classifier import MoodClassifier, create_model


def train_centralized_model(
    total_epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    print_every: int = 5
):
    print("=" * 60)
    print("CENTRALIZED BASELINE TRAINING")
    print("=" * 60)
    print(f"Total Epochs: {total_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 60)
    
    print("\nLoading dataset...")
    platform_data, label_encoder, feature_cols, scaler = load_data(n_platforms=5)
    
    input_dim = len(feature_cols)
    output_dim = len(label_encoder.classes_)
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    print("\nCombining training data from all platforms...")
    X_train_combined = np.vstack([data['X_train'] for data in platform_data])
    y_train_combined = np.concatenate([data['y_train'] for data in platform_data])
    
    X_test = platform_data[0]['X_test']
    y_test = platform_data[0]['y_test']
    
    print(f"Combined training samples: {len(X_train_combined)}")
    print(f"Test samples: {len(X_test)}")
    
    train_dataset = MoodDataset(X_train_combined, y_train_combined)
    test_dataset = MoodDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    model = create_model(input_dim=input_dim, output_dim=output_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    results = {
        'epochs': [],
        'train_losses': [],
        'test_accuracies': [],
        'hyperparameters': {
            'total_epochs': total_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'input_dim': input_dim,
            'output_dim': output_dim
        },
        'dataset_info': {
            'train_samples': len(X_train_combined),
            'test_samples': len(X_test),
            'classes': list(label_encoder.classes_)
        }
    }
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        accuracy = evaluate_model(model, test_loader)
        
        results['epochs'].append(epoch)
        results['train_losses'].append(avg_loss)
        results['test_accuracies'].append(accuracy)
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{total_epochs}: Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%")
    
    final_accuracy = evaluate_model(model, test_loader)
    results['final_accuracy'] = final_accuracy
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    
    return results, model


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy


def save_results(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 3
    TOTAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    PRINT_EVERY = 5
    
    print("Centralized Baseline for Federated Learning Comparison")
    print(f"Equivalent to {NUM_ROUNDS} federated rounds with {LOCAL_EPOCHS} local epochs each")
    print()
    
    results, model = train_centralized_model(
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        print_every=PRINT_EVERY
    )
    
    save_results(results, 'results/centralized_results.json')
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Training Epochs: {TOTAL_EPOCHS}")
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Best Accuracy: {max(results['test_accuracies']):.2f}% (Epoch {results['test_accuracies'].index(max(results['test_accuracies'])) + 1})")
    print(f"Training Samples: {results['dataset_info']['train_samples']}")
    print(f"Test Samples: {results['dataset_info']['test_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
