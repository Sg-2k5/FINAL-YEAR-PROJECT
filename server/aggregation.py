"""
Trust-aware model aggregation module for federated learning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from logging import getLogger

logger = getLogger(__name__)

class TrustAwareAggregator:
    """Implements trust-aware federated averaging for model aggregation."""
    
    def __init__(self, trust_threshold: float = 0.6):
        """Initialize the trust-aware aggregator.
        
        Args:
            trust_threshold: Minimum trust score required for client participation
        """
        self.trust_threshold = trust_threshold
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model updates using trust-weighted federated averaging.
        
        Args:
            server_round: Current round number
            results: List of client results (client, fit_res)
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            return None, {}
        
        # Extract trust scores and weights
        trust_weights = []
        parameters = []
        
        for client, fit_res in results:
            trust_score = fit_res.metrics.get('trust_score', 0.0)
            if trust_score >= self.trust_threshold:
                trust_weights.append(trust_score)
                parameters.append(fit_res.parameters)
        
        if not parameters:
            return None, {}
        
        # Normalize weights
        trust_weights = np.array(trust_weights)
        trust_weights = trust_weights / trust_weights.sum()
        
        # Perform weighted aggregation
        aggregated_params = self._aggregate_parameters(parameters, trust_weights)
        
        metrics = {
            'num_participants': len(parameters),
            'average_trust_score': float(np.mean(trust_weights))
        }
        
        return aggregated_params, metrics
    
    def _aggregate_parameters(
        self,
        parameters: List[fl.common.Parameters],
        weights: np.ndarray
    ) -> fl.common.Parameters:
        """Aggregate parameters using weighted averaging.
        
        Args:
            parameters: List of model parameters from clients
            weights: Trust-based weights for each client
            
        Returns:
            Aggregated parameters
        """
        # Convert parameters to numpy arrays
        params_arrays = [self._parameters_to_numpy(p) for p in parameters]
        
        # Weighted sum of parameters
        weighted_sum = [
            np.sum([w * p[i] for w, p in zip(weights, params_arrays)], axis=0)
            for i in range(len(params_arrays[0]))
        ]
        
        # Convert back to Flower parameters
        return self._numpy_to_parameters(weighted_sum)
    
    def _parameters_to_numpy(self, parameters: fl.common.Parameters) -> List[np.ndarray]:
        """Convert Flower parameters to list of numpy arrays.
        
        Args:
            parameters: Flower parameters
            
        Returns:
            List of numpy arrays
        """
        return [np.frombuffer(p) for p in parameters.tensors]
    
    def _numpy_to_parameters(self, arrays: List[np.ndarray]) -> fl.common.Parameters:
        """Convert list of numpy arrays to Flower parameters.
        
        Args:
            arrays: List of numpy arrays
            
        Returns:
            Flower parameters
        """
        tensors = [arr.tobytes() for arr in arrays]
        return fl.common.Parameters(tensors=tensors, tensor_type="numpy.ndarray")