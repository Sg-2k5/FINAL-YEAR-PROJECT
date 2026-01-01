"""
Trust evaluation engine for federated learning clients.
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import time
from logging import getLogger

logger = getLogger(__name__)

class TrustEngine:
    """Calculates and manages trust scores for federated learning clients."""
    
    def __init__(
        self,
        accuracy_weight: float = 0.30,
        anomaly_weight: float = 0.25,
        participation_weight: float = 0.20,
        history_weight: float = 0.15,
        resource_weight: float = 0.10
    ):
        """Initialize the trust engine.
        
        Args:
            accuracy_weight: Weight for accuracy stability (30%)
            anomaly_weight: Weight for anomaly detection results (25%)
            participation_weight: Weight for participation consistency (20%)
            history_weight: Weight for historical behavior (15%)
            resource_weight: Weight for resource usage patterns (10%)
        """
        self.weights = {
            'accuracy': accuracy_weight,
            'anomaly': anomaly_weight,
            'participation': participation_weight,
            'history': history_weight,
            'resource': resource_weight
        }
        
        # Verify weights sum to 1
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Trust score weights must sum to 1.0")
        
        self.client_history = defaultdict(list)
        self.round_participants = defaultdict(set)
        
    def calculate_trust_score(
        self,
        client_id: str,
        current_metrics: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> float:
        """Calculate trust score for a client based on multiple factors.
        
        Args:
            client_id: Unique identifier for the client
            current_metrics: Current round metrics from the client
            history: Historical metrics and performance data
            
        Returns:
            Trust score between 0 and 1
        """
        # Calculate individual component scores
        accuracy_score = self._evaluate_accuracy_stability(history)
        anomaly_score = self._detect_anomalies(current_metrics, history)
        participation_score = self._evaluate_participation(client_id, history)
        history_score = self._evaluate_historical_behavior(history)
        resource_score = self._evaluate_resource_usage(current_metrics)
        
        # Calculate weighted sum
        trust_score = (
            self.weights['accuracy'] * accuracy_score +
            self.weights['anomaly'] * anomaly_score +
            self.weights['participation'] * participation_score +
            self.weights['history'] * history_score +
            self.weights['resource'] * resource_score
        )
        
        # Ensure score is between 0 and 1
        trust_score = max(0.0, min(1.0, trust_score))
        
        # Update history
        self._update_history(client_id, trust_score, current_metrics)
        
        return trust_score
    
    def _evaluate_accuracy_stability(self, history: List[Dict[str, Any]]) -> float:
        """Evaluate stability of client's accuracy over time.
        
        Args:
            history: List of historical performance metrics
            
        Returns:
            Score between 0 and 1
        """
        if not history:
            return 1.0
        
        # Extract accuracy values from history
        accuracies = [
            h['metrics'].get('accuracy', 0.0)
            for h in history[-5:]  # Look at last 5 rounds
        ]
        
        if not accuracies:
            return 1.0
        
        # Calculate stability score based on variance
        variance = np.var(accuracies) if len(accuracies) > 1 else 0
        stability = 1.0 / (1.0 + variance)
        
        # Consider trend
        if len(accuracies) > 1:
            trend = np.mean(np.diff(accuracies))
            trend_factor = 1.0 + trend  # Reward improving trend
        else:
            trend_factor = 1.0
        
        return float(stability * trend_factor)
    
    def _detect_anomalies(
        self,
        current_metrics: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> float:
        """Detect anomalies in client behavior.
        
        Args:
            current_metrics: Current round metrics
            history: Historical metrics
            
        Returns:
            Score between 0 and 1 (1 = no anomalies)
        """
        if not history:
            return 1.0
        
        anomaly_score = 1.0
        
        # Check training time anomalies
        if 'training_time' in current_metrics:
            times = [h['metrics'].get('training_time', 0) for h in history]
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times) if len(times) > 1 else mean_time
                current_time = current_metrics['training_time']
                
                if std_time > 0:
                    z_score = abs(current_time - mean_time) / std_time
                    time_score = 1.0 / (1.0 + z_score)
                    anomaly_score *= time_score
        
        # Check loss anomalies
        if 'loss' in current_metrics:
            losses = [h['metrics'].get('loss', 0) for h in history]
            if losses:
                mean_loss = np.mean(losses)
                std_loss = np.std(losses) if len(losses) > 1 else mean_loss
                current_loss = current_metrics['loss']
                
                if std_loss > 0:
                    z_score = abs(current_loss - mean_loss) / std_loss
                    loss_score = 1.0 / (1.0 + z_score)
                    anomaly_score *= loss_score
        
        return float(anomaly_score)
    
    def _evaluate_participation(
        self,
        client_id: str,
        history: List[Dict[str, Any]]
    ) -> float:
        """Evaluate client's participation consistency.
        
        Args:
            client_id: Client identifier
            history: Historical participation data
            
        Returns:
            Score between 0 and 1
        """
        if not history:
            return 1.0
        
        total_rounds = max(r['round'] for r in history) + 1
        participation_rate = len(history) / total_rounds
        
        # Consider recent participation more important
        recent_rounds = set(h['round'] for h in history[-5:])
        expected_recent = set(range(total_rounds - 5, total_rounds))
        recent_participation = len(recent_rounds) / min(5, total_rounds)
        
        # Combine overall and recent participation
        participation_score = 0.7 * participation_rate + 0.3 * recent_participation
        
        return float(participation_score)
    
    def _evaluate_historical_behavior(self, history: List[Dict[str, Any]]) -> float:
        """Evaluate client's historical behavior patterns.
        
        Args:
            history: Historical performance data
            
        Returns:
            Score between 0 and 1
        """
        if not history:
            return 1.0
        
        # Calculate long-term performance trend
        accuracies = [h['metrics'].get('accuracy', 0.0) for h in history]
        if len(accuracies) > 1:
            trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            trend_score = 1.0 / (1.0 + np.exp(-10 * trend))  # Sigmoid scaling
        else:
            trend_score = 0.5
        
        # Consider consistency in behavior
        if len(history) > 1:
            metrics_stability = np.mean([
                1.0 / (1.0 + np.std([h['metrics'].get(m, 0.0) for h in history]))
                for m in ['accuracy', 'loss', 'training_time']
                if any(m in h['metrics'] for h in history)
            ])
        else:
            metrics_stability = 1.0
        
        return float(0.4 * trend_score + 0.6 * metrics_stability)
    
    def _evaluate_resource_usage(self, metrics: Dict[str, Any]) -> float:
        """Evaluate client's resource usage patterns.
        
        Args:
            metrics: Current metrics including resource usage
            
        Returns:
            Score between 0 and 1
        """
        resource_score = 1.0
        
        # Check CPU usage
        if 'cpu_percent' in metrics:
            cpu_score = 1.0 - (metrics['cpu_percent'] / 100.0)
            resource_score *= cpu_score
        
        # Check memory usage
        if 'memory_percent' in metrics:
            memory_score = 1.0 - (metrics['memory_percent'] / 100.0)
            resource_score *= memory_score
        
        # Check training time efficiency
        if 'training_time' in metrics and 'batch_size' in metrics:
            time_per_sample = metrics['training_time'] / metrics['batch_size']
            time_score = 1.0 / (1.0 + time_per_sample)
            resource_score *= time_score
        
        return float(resource_score)
    
    def _update_history(
        self,
        client_id: str,
        trust_score: float,
        metrics: Dict[str, Any]
    ):
        """Update client history with new trust score and metrics.
        
        Args:
            client_id: Client identifier
            trust_score: Calculated trust score
            metrics: Current round metrics
        """
        timestamp = time.time()
        self.client_history[client_id].append({
            'timestamp': timestamp,
            'trust_score': trust_score,
            'metrics': metrics
        })