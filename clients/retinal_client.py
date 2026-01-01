"""
Retinal scans client implementation for federated learning.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import time
from logging import getLogger
from torchvision import models

from utils.data_simulation import RetinalDataLoader
from utils.monitoring import DataMonitor, PerformanceMonitor

logger = getLogger(__name__)

class FederatedModel(nn.Module):
    """Standardized neural network model for federated learning across all clients."""
    
    def __init__(self, input_size: int):
        """Initialize the model.
        
        Args:
            input_size: Number of input features
        """
        super().__init__()
        
        # Standardized architecture for all clients (simplified for federated learning)
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Model predictions
        """
        return self.network(x)

class RetinalClient(fl.client.NumPyClient):
    """Federated learning client for retinal scan analysis."""
    
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 1
    ):
        """Initialize the client.
        
        Args:
            data_dir: Directory containing the data
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            local_epochs: Number of local training epochs
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        # Initialize data loader
        self.data_loader = RetinalDataLoader(data_dir)
        
        # In a real implementation, you would need to:
        # 1. Download APTOS dataset from Kaggle
        # 2. Extract and organize images
        # Load processed retinal features
        X, y = self.data_loader.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1)  # Binary classification
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        
        # Initialize model with standardized architecture (8 features like blood client)
        STANDARDIZED_INPUT_SIZE = 8
        self.model = FederatedModel(input_size=STANDARDIZED_INPUT_SIZE)
        
        # Add feature transformation to convert retinal features to standardized size
        self.feature_transform = nn.Linear(X.shape[1], STANDARDIZED_INPUT_SIZE)
        
        self.criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.feature_transform.parameters()),
            lr=learning_rate
        )
        
        # Initialize monitors
        self.data_monitor = DataMonitor()
        self.performance_monitor = PerformanceMonitor()
    
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Get model parameters.
        
        Args:
            config: Configuration parameters
            
        Returns:
            List of model parameter arrays
        """
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters.
        
        Args:
            parameters: List of model parameter arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on local data.
        
        Args:
            parameters: Current model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        self.performance_monitor.start_monitoring()
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Training loop
        self.model.train()
        for _ in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # Transform features to standardized dimension
                transformed_data = self.feature_transform(data)
                output = self.model(transformed_data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Update performance metrics
                self.performance_monitor.update_metrics()
        
        # Calculate metrics
        performance_metrics = self.performance_monitor.stop_monitoring()
        
        # Get updated parameters
        parameters_prime = self.get_parameters(config)
        
        # Calculate training metrics
        num_examples = len(self.train_loader.dataset)
        
        metrics = {
            'train_loss': float(loss.item()),
            'num_examples': num_examples,
            'batch_size': self.batch_size,
            **performance_metrics
        }
        
        return parameters_prime, num_examples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        
        # Evaluation loop
        self.model.eval()
        loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # Transform features to standardized dimension
                transformed_data = self.feature_transform(data)
                output = self.model(transformed_data)
                loss += self.criterion(output, target).item()
                pred = (output > 0.5).float()  # Binary prediction
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate metrics
        num_examples = len(self.test_loader.dataset)
        loss /= len(self.test_loader)
        accuracy = correct / num_examples
        
        metrics = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_examples': num_examples
        }
        
        return loss, num_examples, metrics
    
    def start(self):
        """Start the client and connect to the server."""
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8085",
            client=self
        )

if __name__ == "__main__":
    print("👁️ Starting Retinal Scan Client...")
    print("📡 Connecting to federated server at 127.0.0.1:8085")
    client = RetinalClient()
    client.start()