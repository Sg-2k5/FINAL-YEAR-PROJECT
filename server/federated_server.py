"""
Core federated learning server implementation using Flower framework.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import flwr as fl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from logging import getLogger
from pathlib import Path
import json
import datetime
from server.trust_engine import TrustEngine
from server.aggregation import TrustAwareAggregator


logger = getLogger(__name__)

class FederatedServer:
    """Central federated learning server that coordinates training rounds."""
    
    def __init__(
        self,
        port: Optional[int] = None,
        min_clients: int = 3,
        min_available_clients: int = 3,
        trust_threshold: float = 0.6,
        num_rounds: int = 100
    ):
        """Initialize the federated server."""
        # Default port if not provided
        self.port = port or 8080
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        self.trust_threshold = trust_threshold
        self.num_rounds = num_rounds
        
        self.trust_engine = TrustEngine()
        self.aggregator = TrustAwareAggregator(trust_threshold)
        self.client_metrics: Dict[str, Dict] = {}
        
        # Initialize activity log for dashboard
        self.activity_log_path = "federated_activity.json"
        self.activity_log = []
    
    def log_activity(self, activity_type: str, message: str, client_id: str = None, round_num: int = None):
        """Log client activity for dashboard monitoring."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': activity_type,
            'message': message,
            'client_id': client_id,
            'round': round_num
        }
        self.activity_log.append(log_entry)
        
        # Save to file for dashboard to read
        try:
            with open(self.activity_log_path, 'w') as f:
                json.dump(self.activity_log, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write activity log: {e}")
    def client_manager(self) -> fl.server.ClientManager:
        """Create a client manager for Flower server.
        
        Returns:
            Configured client manager instance
        """
        return fl.server.SimpleClientManager()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model updates from clients with trust-based filtering.
        
        Args:
            server_round: Current round number
            results: List of client results
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Log round start
        self.log_activity("round_start", f"Starting training round {server_round}", round_num=server_round)
        # Update trust scores based on client performance
        for client, fit_res in results:
            client_id = client.cid
            metrics = fit_res.metrics
            
            # Log client participation
            self.log_activity("client_fit", f"Client {client_id} completed training", 
                            client_id=client_id, round_num=server_round)
            
            # Update client metrics history
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = {
                    'history': [],
                    'trust_score': 1.0  # Initial trust score
                }
                self.log_activity("client_connect", f"New client {client_id} connected", 
                                client_id=client_id, round_num=server_round)
            
            self.client_metrics[client_id]['history'].append({
                'round': server_round,
                'metrics': metrics
            })
            
            # Calculate new trust score
            trust_score = self.trust_engine.calculate_trust_score(
                client_id,
                metrics,
                self.client_metrics[client_id]['history']
            )
            self.client_metrics[client_id]['trust_score'] = trust_score
            
            # Log trust score update
            self.log_activity("trust_update", f"Client {client_id} trust score: {trust_score:.3f}", 
                            client_id=client_id, round_num=server_round)
        
        # Filter clients based on trust scores
        trusted_results = [
            (client, res) for client, res in results
            if self.client_metrics[client.cid]['trust_score'] >= self.trust_threshold
        ]
        
        if not trusted_results:
            return None, {}
        
        # Aggregate parameters from trusted clients
        return self.aggregator.aggregate_fit(server_round, trusted_results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: List of client evaluation results
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_metrics, metrics_dict)
        """
        if not results:
            return None, {}
        
        # Aggregate metrics from trusted clients only
        # Initialize client metrics if not present
        for client, res in results:
            if client.cid not in self.client_metrics:
                self.client_metrics[client.cid] = {
                    'trust_score': 1.0,  # Default trust score for new clients
                    'performance_history': [],
                    'accuracy_history': []
                }
        
        trusted_results = [
            (client, res) for client, res in results
            if self.client_metrics[client.cid]['trust_score'] >= self.trust_threshold
        ]
        
        # Calculate weighted average of metrics based on trust scores
        weighted_metrics = []
        trust_weights = []
        
        for client, res in trusted_results:
            trust_score = self.client_metrics[client.cid]['trust_score']
            weighted_metrics.append(res.loss * trust_score)
            trust_weights.append(trust_score)
        
        if not trust_weights:
            return None, {}
        
        aggregated_loss = sum(weighted_metrics) / sum(trust_weights)
        
        metrics = {
            "aggregated_loss": aggregated_loss,
            "num_trusted_clients": len(trusted_results),
            "total_clients": len(results)
        }
        
        return aggregated_loss, metrics

    def start(self, port: Optional[int] = None):
        """Start the federated learning server."""
        server_port = port or self.port  # Use runtime port if provided

        # Define custom strategy class
        class TrustAwareStrategy(fl.server.strategy.FedAvg):
            def __init__(self, server_instance):
                super().__init__(
                    fraction_fit=1.0,
                    fraction_evaluate=1.0,
                    min_fit_clients=server_instance.min_clients,
                    min_evaluate_clients=server_instance.min_clients,
                    min_available_clients=server_instance.min_available_clients,
                    on_fit_config_fn=server_instance.fit_config,
                    on_evaluate_config_fn=server_instance.evaluate_config
                )
                self.server = server_instance
                
            def aggregate_fit(self, server_round, results, failures):
                return self.server.aggregate_fit(server_round, results, failures)
                
            def aggregate_evaluate(self, server_round, results, failures):
                return self.server.aggregate_evaluate(server_round, results, failures)
        
        strategy = TrustAwareStrategy(self)

        # Start server
        fl.server.start_server(
            server_address=f"0.0.0.0:{server_port}",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy
        )

    def fit_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return training configuration for clients.
        
        Args:
            server_round: Current round number
            
        Returns:
            Training configuration dictionary
        """
        return {
            "server_round": server_round,
            "local_epochs": 1,
            "batch_size": 32
        }
    
    def evaluate_config(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return evaluation configuration for clients.
        
        Args:
            server_round: Current round number
            
        Returns:
            Evaluation configuration dictionary
        """
        return {
            "server_round": server_round,
            "batch_size": 32
        }
    
    def save_metrics(self, path: Union[str, Path]):
        """Save client metrics and trust scores to file.
        
        Args:
            path: Path to save the metrics
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.client_metrics, f, indent=4)
    
    def load_metrics(self, path: Union[str, Path]):
        """Load client metrics and trust scores from file.
        
        Args:
            path: Path to load the metrics from
        """
        path = Path(path)
        if path.exists():
            with open(path, 'r') as f:
                self.client_metrics = json.load(f)

if __name__ == "__main__":
    print("🚀 Starting Trust-Aware Federated Learning Server...")
    server = FederatedServer(
        port=8085,
        num_rounds=25,  # Increased rounds for better dashboard logging
        min_clients=3,  # Wait for all 3 clients
        min_available_clients=3,  # Ensure all 3 clients participate
        trust_threshold=0.5
    )
    print("📡 Server listening on port 8085")
    print("⏳ Waiting for all 3 clients to connect...")
    server.start()