"""
Admin dashboard interface for federated learning system monitoring.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import time
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta

from security.blockchain_audit import BlockchainAuditor
from security.honeypot import HoneypotManager
from utils.monitoring import DataMonitor, PerformanceMonitor

class AdminDashboard:
    """Admin dashboard for monitoring federated learning system."""
    
    def __init__(
        self,
        blockchain_provider: str = "http://localhost:8545",
        contract_address: str = None,
        db_path: str = "security.db"
    ):
        """Initialize the dashboard.
        
        Args:
            blockchain_provider: URL of the blockchain provider
            contract_address: Address of the audit contract
            db_path: Path to security database
        """
        self.auditor = BlockchainAuditor(
            provider_url=blockchain_provider,
            contract_address=contract_address
        )
        self.honeypot = HoneypotManager(db_path=db_path)
        
        # Cache for performance
        self.cache_timeout = 60  # seconds
        self.cached_data = {}
    
    def render_dashboard(self):
        """Render the main dashboard interface."""
        st.title("Federated Learning Admin Dashboard")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Client Monitor", "Security", "Performance", "Audit Logs"]
        )
        
        if page == "Overview":
            self._render_overview()
        elif page == "Client Monitor":
            self._render_client_monitor()
        elif page == "Security":
            self._render_security_page()
        elif page == "Performance":
            self._render_performance_page()
        elif page == "Audit Logs":
            self._render_audit_logs()
    
    def _render_overview(self):
        """Render system overview page."""
        st.header("System Overview")
        
        # System status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Clients",
                value=self._get_active_clients_count()
            )
        
        with col2:
            st.metric(
                label="Training Round",
                value=self._get_current_round()
            )
        
        with col3:
            st.metric(
                label="Global Model Accuracy",
                value=f"{self._get_global_accuracy():.2f}%"
            )
        
        with col4:
            st.metric(
                label="Security Incidents (24h)",
                value=self._get_security_incident_count()
            )
        
        # Training progress chart
        st.subheader("Training Progress")
        progress_df = self._get_training_progress()
        fig = px.line(
            progress_df,
            x="round",
            y=["accuracy", "loss"],
            title="Model Performance Over Time"
        )
        st.plotly_chart(fig)
        
        # Client trust scores
        st.subheader("Client Trust Scores")
        trust_df = self._get_client_trust_scores()
        fig = px.bar(
            trust_df,
            x="client_id",
            y="trust_score",
            color="status",
            title="Client Trust Distribution"
        )
        st.plotly_chart(fig)
    
    def _render_client_monitor(self):
        """Render client monitoring page."""
        st.header("Client Monitor")
        
        # Client selector
        client_id = st.selectbox(
            "Select Client",
            options=self._get_all_clients()
        )
        
        if client_id:
            # Client details
            st.subheader("Client Details")
            client_info = self._get_client_info(client_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Status:", client_info['status'])
                st.write("Trust Score:", f"{client_info['trust_score']:.3f}")
                st.write("Last Active:", client_info['last_active'])
            
            with col2:
                st.write("Data Type:", client_info['data_type'])
                st.write("Model Type:", client_info['model_type'])
                st.write("Updates Contributed:", client_info['updates_count'])
            
            # Performance metrics
            st.subheader("Performance Metrics")
            metrics_df = self._get_client_metrics(client_id)
            fig = px.line(
                metrics_df,
                x="timestamp",
                y=["accuracy", "loss", "training_time"],
                title="Client Performance Metrics"
            )
            st.plotly_chart(fig)
            
            # Resource usage
            st.subheader("Resource Usage")
            resource_df = self._get_resource_usage(client_id)
            fig = px.line(
                resource_df,
                x="timestamp",
                y=["cpu_percent", "memory_percent"],
                title="Resource Utilization"
            )
            st.plotly_chart(fig)
    
    def _render_security_page(self):
        """Render security monitoring page."""
        st.header("Security Monitor")
        
        # Security overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Quarantined Clients",
                value=self._get_quarantined_count()
            )
        
        with col2:
            st.metric(
                label="Active Threats",
                value=self._get_active_threats_count()
            )
        
        with col3:
            st.metric(
                label="Decoy Accesses",
                value=self._get_decoy_access_count()
            )
        
        # Security incidents timeline
        st.subheader("Security Incidents Timeline")
        incidents_df = self._get_security_incidents()
        fig = px.scatter(
            incidents_df,
            x="timestamp",
            y="severity",
            color="event_type",
            size="impact",
            hover_data=["client_id", "details"],
            title="Security Incidents"
        )
        st.plotly_chart(fig)
        
        # Threat analysis
        st.subheader("Threat Analysis")
        threat_df = self._get_threat_analysis()
        fig = px.pie(
            threat_df,
            values="count",
            names="threat_type",
            title="Threat Distribution"
        )
        st.plotly_chart(fig)
    
    def _render_performance_page(self):
        """Render system performance monitoring page."""
        st.header("System Performance")
        
        # Performance metrics over time
        st.subheader("System Metrics")
        perf_df = self._get_system_performance()
        fig = px.line(
            perf_df,
            x="timestamp",
            y=["cpu_usage", "memory_usage", "network_io"],
            title="System Resource Usage"
        )
        st.plotly_chart(fig)
        
        # Training statistics
        st.subheader("Training Statistics")
        stats_df = self._get_training_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                stats_df,
                y="round_time",
                title="Training Round Times"
            )
            st.plotly_chart(fig)
        
        with col2:
            fig = px.histogram(
                stats_df,
                x="client_count",
                title="Active Clients per Round"
            )
            st.plotly_chart(fig)
    
    def _render_audit_logs(self):
        """Render blockchain audit logs page."""
        st.header("Audit Logs")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        if start_date and end_date:
            # Fetch audit logs
            logs = self._get_audit_logs(start_date, end_date)
            
            if not logs.empty:
                st.write(f"Found {len(logs)} audit log entries")
                
                # Display logs with better formatting
                if 'data' in logs.columns:
                    # Expand the data column for better readability
                    display_logs = logs.copy()
                    display_logs['data'] = display_logs['data'].astype(str)
                
                st.dataframe(display_logs)
                
                # Export button
                if st.button("Export Logs"):
                    self._export_logs(logs)
            else:
                st.info("No audit logs found for the selected date range.")
                st.write("💡 **Note:** The system is running in mock blockchain mode. Audit logs are simulated for demonstration purposes.")
    
    def _get_active_clients_count(self) -> int:
        """Get number of currently active clients."""
        # Implementation would fetch actual data
        return 3
    
    def _get_current_round(self) -> int:
        """Get current training round number."""
        # Implementation would fetch actual data
        return 42
    
    def _get_global_accuracy(self) -> float:
        """Get current global model accuracy."""
        # Implementation would fetch actual data
        return 95.5
    
    def _get_security_incident_count(self) -> int:
        """Get number of security incidents in last 24 hours."""
        return self.honeypot.get_security_report(24 * 60 * 60)['event_counts'].get('critical', 0)
    
    def _get_training_progress(self) -> pd.DataFrame:
        """Get training progress data."""
        # Implementation would fetch actual data
        rounds = range(1, 43)
        return pd.DataFrame({
            'round': rounds,
            'accuracy': [85 + np.random.normal(0, 1) for _ in rounds],
            'loss': [0.3 * np.exp(-0.05 * r) + np.random.normal(0, 0.01) for r in rounds]
        })
    
    def _get_client_trust_scores(self) -> pd.DataFrame:
        """Get current trust scores for all clients."""
        # Implementation would fetch actual data
        return pd.DataFrame({
            'client_id': ['client1', 'client2', 'client3'],
            'trust_score': [0.95, 0.85, 0.75],
            'status': ['normal', 'normal', 'suspicious']
        })
    
    def _get_all_clients(self) -> List[str]:
        """Get list of all client IDs."""
        # Implementation would fetch actual data
        return ['client1', 'client2', 'client3']
    
    def _get_client_info(self, client_id: str) -> Dict[str, Any]:
        """Get detailed information about a client."""
        # Implementation would fetch actual data
        return {
            'status': 'active',
            'trust_score': 0.95,
            'last_active': '2025-09-25 10:30:00',
            'data_type': 'blood_reports',
            'model_type': 'neural_network',
            'updates_count': 42
        }
    
    def _get_client_metrics(self, client_id: str) -> pd.DataFrame:
        """Get performance metrics for a client."""
        # Implementation would fetch actual data
        timestamps = pd.date_range(
            start='2025-09-24',
            end='2025-09-25',
            periods=24
        )
        return pd.DataFrame({
            'timestamp': timestamps,
            'accuracy': [90 + np.random.normal(0, 1) for _ in timestamps],
            'loss': [0.2 + np.random.normal(0, 0.01) for _ in timestamps],
            'training_time': [10 + np.random.normal(0, 0.5) for _ in timestamps]
        })
    
    def _get_resource_usage(self, client_id: str) -> pd.DataFrame:
        """Get resource usage metrics for a client."""
        # Implementation would fetch actual data
        timestamps = pd.date_range(
            start='2025-09-24',
            end='2025-09-25',
            periods=24
        )
        return pd.DataFrame({
            'timestamp': timestamps,
            'cpu_percent': [30 + np.random.normal(0, 5) for _ in timestamps],
            'memory_percent': [40 + np.random.normal(0, 3) for _ in timestamps]
        })
    
    def _get_quarantined_count(self) -> int:
        """Get number of quarantined clients."""
        return self.honeypot.get_security_report()['quarantined_clients']
    
    def _get_active_threats_count(self) -> int:
        """Get number of active security threats."""
        # Implementation would fetch actual data
        return 2
    
    def _get_decoy_access_count(self) -> int:
        """Get number of decoy endpoint accesses."""
        return self.honeypot.get_security_report()['decoy_accesses']
    
    def _get_security_incidents(self) -> pd.DataFrame:
        """Get security incidents data."""
        # Implementation would fetch actual data
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2025-09-24', periods=10, freq='H'),
            'severity': ['high', 'medium', 'low'] * 3 + ['critical'],
            'event_type': ['anomaly', 'decoy_access', 'poisoning'] * 3 + ['breach'],
            'impact': np.random.randint(1, 10, 10),
            'client_id': [f'client{i%3 + 1}' for i in range(10)],
            'details': ['Suspicious activity detected'] * 10
        })
    
    def _get_threat_analysis(self) -> pd.DataFrame:
        """Get threat analysis data."""
        # Implementation would fetch actual data
        return pd.DataFrame({
            'threat_type': ['Poisoning', 'Model Theft', 'Data Leakage', 'DoS'],
            'count': [15, 8, 5, 3]
        })
    
    def _get_system_performance(self) -> pd.DataFrame:
        """Get system performance metrics."""
        # Implementation would fetch actual data
        timestamps = pd.date_range(
            start='2025-09-24',
            end='2025-09-25',
            periods=24
        )
        return pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': [50 + np.random.normal(0, 5) for _ in timestamps],
            'memory_usage': [60 + np.random.normal(0, 3) for _ in timestamps],
            'network_io': [40 + np.random.normal(0, 4) for _ in timestamps]
        })
    
    def _get_training_stats(self) -> pd.DataFrame:
        """Get training statistics data."""
        # Implementation would fetch actual data
        return pd.DataFrame({
            'round_time': np.random.normal(60, 10, 100),
            'client_count': np.random.randint(2, 4, 100)
        })
    
    def _get_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get blockchain audit logs."""
        logs = self.auditor.get_events(
            from_block=0  # In production, calculate block based on date
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # Handle missing or different timestamp formats
        if df.empty:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['event_type', 'data', 'timestamp', 'block_number', 'transaction_hash'])
        
        if 'timestamp' in df.columns:
            # Convert Unix timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            # If no timestamp column, create one with current time
            df['timestamp'] = pd.Timestamp.now()
        
        # Filter by date range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        return df[mask]
    
    def _export_logs(self, logs: pd.DataFrame):
        """Export audit logs to file."""
        logs.to_csv(f"audit_logs_{int(time.time())}.csv", index=False)
        st.success("Logs exported successfully!")

def launch_dashboard():
    """Launch the admin dashboard."""
    dashboard = AdminDashboard()
    dashboard.render_dashboard()