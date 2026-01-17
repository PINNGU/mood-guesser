import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import requests
import numpy as np

from src.models.mood_classifier import create_model
from src.utils.dataset_loader import MoodDataset


class PlatformClient:
    def __init__(self, platform_id, X_train, y_train, X_test, y_test, 
                 coordinator_urls, input_dim=7, output_dim=5):
        self.platform_id = platform_id
        self.coordinator_urls = coordinator_urls
        self.current_coordinator_idx = 0
        
        # Create datasets and data loaders
        self.train_dataset = MoodDataset(X_train, y_train)
        self.test_dataset = MoodDataset(X_test, y_test)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        # Model, optimizer, and loss function
        self.model = create_model(input_dim, output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.local_rounds = 0
        self.global_model_version = 0
    
    def get_coordinator_url(self):
        """Returns current coordinator URL."""
        return self.coordinator_urls[self.current_coordinator_idx]
    
    def try_next_coordinator(self):
        """Cycles to next coordinator index."""
        self.current_coordinator_idx = (self.current_coordinator_idx + 1) % len(self.coordinator_urls)
    
    def train_local_epoch(self, epochs=1):
        """Trains model locally for specified number of epochs."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        self.local_rounds += 1
        average_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return average_loss
    
    def evaluate(self):
        """Evaluates model on test set and returns accuracy percentage."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return accuracy
    
    def get_model_update(self):
        """Returns model update dict with serialized state."""
        state_dict = self.model.get_state_dict()
        serialized_state = {
            key: value.cpu().numpy().tolist() 
            for key, value in state_dict.items()
        }
        
        return {
            'platform_id': self.platform_id,
            'model_state': serialized_state,
            'n_samples': len(self.train_dataset),
            'local_rounds': self.local_rounds,
            'model_version': self.global_model_version
        }
    
    def send_update_to_coordinator(self, model_update):
        """Sends POST request to coordinator with model update."""
        num_coordinators = len(self.coordinator_urls)
        
        for attempt in range(num_coordinators):
            coordinator_url = self.get_coordinator_url()
            url = f"{coordinator_url}/platform/update"
            
            try:
                response = requests.post(url, json=model_update, timeout=30)
                response.raise_for_status()
                return response
            except (requests.RequestException, Exception) as e:
                print(f"Failed to connect to coordinator {coordinator_url}: {e}")
                self.try_next_coordinator()
        
        return None
    
    def receive_global_model(self, global_model_state):
        """Updates local model with global model state."""
        # Convert lists back to tensors
        tensor_state = {
            key: torch.tensor(value) 
            for key, value in global_model_state.items()
        }
        
        self.model.load_state_dict_from_dict(tensor_state)
        self.global_model_version += 1
    
    def run_training_round(self):
        """Executes a complete training round."""
        # Train local epoch
        avg_loss = self.train_local_epoch()
        print(f"Platform {self.platform_id}: Local training complete, avg loss: {avg_loss:.4f}")
        
        # Get model update
        model_update = self.get_model_update()
        print(f"Platform {self.platform_id}: Model update prepared with {model_update['n_samples']} samples")
        
        # Send to coordinator
        response = self.send_update_to_coordinator(model_update)
        if response is not None:
            print(f"Platform {self.platform_id}: Update sent successfully to coordinator")
        else:
            print(f"Platform {self.platform_id}: Failed to send update to any coordinator")
        
        return response
