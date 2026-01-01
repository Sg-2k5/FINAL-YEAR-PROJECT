"""
Monitoring utilities for tracking dataset statistics and data quality.
"""

import logging
import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import roc_auc_score
import psutil
import time

logger = logging.getLogger(__name__)

class DataMonitor:
    """Monitor data quality and statistics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for numerical data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary of computed statistics
        """
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'missing_ratio': float(np.isnan(data).mean())
        }
    
    def check_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Check data quality metrics.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary of quality metrics
        """
        return {
            'completeness': 1 - np.isnan(data).mean(),
            'value_range': (float(np.min(data)), float(np.max(data))),
            'outliers_ratio': self._detect_outliers(data)
        }
    
    def _detect_outliers(self, data: np.ndarray) -> float:
        """Detect outliers using IQR method.
        
        Args:
            data: Input data array
            
        Returns:
            Ratio of outliers in the data
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outlier_mask = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        return float(outlier_mask.mean())

class PerformanceMonitor:
    """Monitor system performance during data processing."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start monitoring system performance."""
        self.start_time = time.time()
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'processing_time': None
        }
    
    def update_metrics(self):
        """Update current performance metrics."""
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_percent'].append(psutil.Process().memory_percent())
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return final metrics.
        
        Returns:
            Dictionary of performance metrics (scalar values only for federated learning)
        """
        if self.start_time is None:
            raise RuntimeError("Monitoring was not started")
        
        # Return only scalar values that Flower can serialize
        return {
            'processing_time': float(time.time() - self.start_time),
            'avg_cpu_percent': float(np.mean(self.metrics['cpu_percent'])),
            'avg_memory_percent': float(np.mean(self.metrics['memory_percent'])),
            'max_cpu_percent': float(np.max(self.metrics['cpu_percent'])),
            'max_memory_percent': float(np.max(self.metrics['memory_percent']))
        }

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        # For binary classification
        auc = roc_auc_score(y_true, y_pred)
        accuracy = np.mean(y_true == (y_pred > 0.5))
        
        return {
            'auc': float(auc),
            'accuracy': float(accuracy)
        }
    except Exception as e:
        logger.warning(f"Could not compute all metrics: {str(e)}")
        return {
            'accuracy': float(np.mean(y_true == y_pred))
        }