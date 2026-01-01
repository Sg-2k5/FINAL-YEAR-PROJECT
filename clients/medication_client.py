"""
Medication history client implementation for federated learning.
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

from utils.data_simulation import MedicationDataLoader
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

class MedicationLSTM(nn.Module):
    """LSTM model for medication history analysis."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """Initialize the model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary classification: effective/ineffective
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Model predictions
        """
        lstm_out, _ = self.lstm(x)
        
        # Use last sequence output for classification
        last_output = lstm_out[:, -1, :]
        
        return self.fc(last_output)

class MedicationClient(fl.client.NumPyClient):
    """Federated learning client for medication history analysis."""
    
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
        self.data_loader = MedicationDataLoader(data_dir)
        X, y = self.data_loader.load_data()  # Now returns sequences and labels directly
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Flatten sequences for standardized model (convert from 3D to 2D)
        # Reshape from (batch, seq_len, features) to (batch, seq_len * features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train_flat)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.X_test = torch.FloatTensor(X_test_flat)
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
        flattened_size = X.shape[1] * X.shape[2]  # seq_length * n_features  
        self.model = FederatedModel(input_size=STANDARDIZED_INPUT_SIZE)
        
        # Add feature transformation to convert medication features to standardized size
        self.feature_transform = nn.Linear(flattened_size, STANDARDIZED_INPUT_SIZE)
        
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
                # Flatten and transform features to standardized dimension
                flattened_data = data.view(data.size(0), -1)
                transformed_data = self.feature_transform(flattened_data)
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
                # Flatten and transform features to standardized dimension
                flattened_data = data.view(data.size(0), -1)
                transformed_data = self.feature_transform(flattened_data)
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
    print("💊 Starting Medication History Client...")
    print("📡 Connecting to federated server at 127.0.0.1:8085")
    client = MedicationClient()
    client.start()