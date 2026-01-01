"""
Anomaly detection module for federated learning security.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
from typing import Dict, List, Tuple, Any
import time
from logging import getLogger

logger = getLogger(__name__)

class AnomalyDetector:
    """Detects anomalies in client behavior and model updates."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        window_size: int = 10,
        threshold_std: float = 3.0
    ):
        """Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            window_size: Size of sliding window for time series analysis
            threshold_std: Number of standard deviations for statistical thresholds
        """
        self.contamination = contamination
        self.window_size = window_size
        self.threshold_std = threshold_std
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.history: Dict[str, List[Dict[str, Any]]] = {}
    
    def detect_time_anomalies(
        self,
        client_id: str,
        training_time: float,
        batch_size: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect anomalies in training time.
        
        Args:
            client_id: Identifier of the client
            training_time: Time taken for training
            batch_size: Number of samples processed
            
        Returns:
            Tuple of (is_anomaly, details)
        """
        if client_id not in self.history:
            self.history[client_id] = []
        
        # Calculate time per sample
        time_per_sample = training_time / batch_size
        
        # Get historical times
        historical_times = [
            h['time_per_sample']
            for h in self.history[client_id][-self.window_size:]
        ]
        
        if len(historical_times) >= 2:
            mean_time = np.mean(historical_times)
            std_time = np.std(historical_times)
            
            # Check for statistical anomaly
            z_score = abs(time_per_sample - mean_time) / std_time
            is_anomaly = z_score > self.threshold_std
            
            details = {
                'z_score': float(z_score),
                'threshold': float(self.threshold_std),
                'historical_mean': float(mean_time),
                'historical_std': float(std_time)
            }
        else:
            is_anomaly = False
            details = {'message': 'Insufficient history'}
        
        # Update history
        self.history[client_id].append({
            'timestamp': time.time(),
            'time_per_sample': time_per_sample
        })
        
        return is_anomaly, details
    
    def detect_update_anomalies(
        self,
        model_update: List[np.ndarray],
        previous_updates: List[List[np.ndarray]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect anomalies in model updates.
        
        Args:
            model_update: Current model update parameters
            previous_updates: List of previous model updates
            
        Returns:
            Tuple of (is_anomaly, details)
        """
        # Flatten updates for analysis
        flat_update = np.concatenate([arr.flatten() for arr in model_update])
        flat_previous = np.array([
            np.concatenate([arr.flatten() for arr in update])
            for update in previous_updates
        ])
        
        if len(flat_previous) > 0:
            # Fit isolation forest
            self.isolation_forest.fit(flat_previous.reshape(-1, 1))
            
            # Predict anomaly
            anomaly_score = self.isolation_forest.score_samples(
                flat_update.reshape(1, -1)
            )[0]
            
            # Lower scores indicate more anomalous samples
            is_anomaly = anomaly_score < -0.5  # Threshold can be adjusted
            
            details = {
                'anomaly_score': float(anomaly_score),
                'magnitude': float(np.linalg.norm(flat_update)),
                'mean_magnitude': float(np.mean([
                    np.linalg.norm(update) for update in flat_previous
                ]))
            }
        else:
            is_anomaly = False
            details = {'message': 'Insufficient history'}
        
        return is_anomaly, details
    
    def detect_behavior_anomalies(
        self,
        metrics: Dict[str, Any],
        client_id: str
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect anomalies in client behavior metrics.
        
        Args:
            metrics: Dictionary of client metrics
            client_id: Identifier of the client
            
        Returns:
            Tuple of (is_anomaly, anomaly_scores)
        """
        if client_id not in self.history:
            self.history[client_id] = []
        
        # Extract relevant metrics
        relevant_metrics = [
            metrics.get('accuracy', 0.0),
            metrics.get('loss', 0.0),
            metrics.get('training_time', 0.0),
            metrics.get('cpu_percent', 0.0),
            metrics.get('memory_percent', 0.0)
        ]
        
        # Get historical metrics
        historical_metrics = [
            [
                h.get('accuracy', 0.0),
                h.get('loss', 0.0),
                h.get('training_time', 0.0),
                h.get('cpu_percent', 0.0),
                h.get('memory_percent', 0.0)
            ]
            for h in self.history[client_id][-self.window_size:]
        ]
        
        if len(historical_metrics) >= 2:
            # Scale metrics
            scaled_metrics = self.scaler.fit_transform(
                np.array(historical_metrics + [relevant_metrics])
            )
            
            # Last row contains current metrics
            current_scaled = scaled_metrics[-1]
            historical_scaled = scaled_metrics[:-1]
            
            # Calculate Mahalanobis distance
            mean = np.mean(historical_scaled, axis=0)
            cov = np.cov(historical_scaled.T)
            inv_covmat = np.linalg.inv(cov)
            
            mahalanobis_dist = np.sqrt(
                (current_scaled - mean).dot(inv_covmat).dot(
                    (current_scaled - mean).T
                )
            )
            
            # Check for anomaly
            is_anomaly = mahalanobis_dist > self.threshold_std
            
            anomaly_scores = {
                'mahalanobis_distance': float(mahalanobis_dist),
                'threshold': float(self.threshold_std)
            }
        else:
            is_anomaly = False
            anomaly_scores = {'message': 'Insufficient history'}
        
        # Update history
        self.history[client_id].append(metrics)
        
        return is_anomaly, anomaly_scores
    
    def detect_poisoning_attacks(
        self,
        model_parameters: List[np.ndarray],
        previous_parameters: List[List[np.ndarray]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect potential poisoning attacks in model updates.
        
        Args:
            model_parameters: Current model parameters
            previous_parameters: List of previous model parameters
            
        Returns:
            Tuple of (is_attack, details)
        """
        if not previous_parameters:
            return False, {'message': 'Insufficient history'}
        
        # Calculate cosine similarity with previous updates
        current_update = np.concatenate([p.flatten() for p in model_parameters])
        
        similarities = []
        for prev_params in previous_parameters:
            prev_update = np.concatenate([p.flatten() for p in prev_params])
            similarity = np.dot(current_update, prev_update) / (
                np.linalg.norm(current_update) * np.linalg.norm(prev_update)
            )
            similarities.append(similarity)
        
        # Check for suspicious patterns
        mean_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        is_attack = (
            mean_similarity < 0.1 or  # Very different from all previous updates
            min_similarity < -0.5     # Strong negative correlation
        )
        
        details = {
            'mean_similarity': float(mean_similarity),
            'min_similarity': float(min_similarity),
            'suspicion_level': 'high' if is_attack else 'low'
        }
        
        return is_attack, details