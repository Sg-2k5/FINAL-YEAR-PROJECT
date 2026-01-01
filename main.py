"""
Trust-Aware Vertical Federated Learning Framework
Main entry point for the federated learning system.
"""

import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def start_server():
    from server.federated_server import FederatedServer
    server = FederatedServer(port=8080)  # 👈 set your port
    server.start()

def start_client(client_type):
    """Start a client node of the specified type."""
    if client_type == "blood":
        from clients.blood_client import BloodClient
        client = BloodClient()
    elif client_type == "retinal":
        from clients.retinal_client import RetinalClient
        client = RetinalClient()
    elif client_type == "medication":
        from clients.medication_client import MedicationClient
        client = MedicationClient()
    else:
        raise ValueError(f"Unknown client type: {client_type}")
    
    client.start()

def start_dashboard():
    """Launch the admin dashboard."""
    from dashboard.admin_interface import launch_dashboard
    launch_dashboard()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Trust-Aware Federated Learning System")
    parser.add_argument("mode", choices=["server", "client", "dashboard"],
                      help="Operation mode: server, client, or dashboard")
    parser.add_argument("--type", choices=["blood", "retinal", "medication"],
                      help="Client type (required for client mode)")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        if args.mode == "server":
            start_server()
        elif args.mode == "client":
            if not args.type:
                parser.error("Client type is required for client mode")
            start_client(args.type)
        elif args.mode == "dashboard":
            start_dashboard()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()