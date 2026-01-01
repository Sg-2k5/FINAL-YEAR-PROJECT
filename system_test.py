#!/usr/bin/env python3
"""
Quick system test to verify all components are working.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all critical imports work."""
    try:
        print("Testing imports...")
        
        # Test federated learning framework
        import flwr as fl
        print("✓ Flower (flwr) import successful")
        
        # Test ML frameworks
        import torch
        print("✓ PyTorch import successful")
        
        import tensorflow as tf
        print("✓ TensorFlow import successful")
        
        import sklearn
        print("✓ Scikit-learn import successful")
        
        # Test project modules
        from utils.data_simulation import BloodDataLoader
        print("✓ BloodDataLoader import successful")
        
        from security.blockchain_audit import BlockchainAuditor
        print("✓ BlockchainAuditor import successful")
        
        from server.trust_engine import TrustEngine
        print("✓ TrustEngine import successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_blockchain_mock():
    """Test blockchain auditor in mock mode."""
    try:
        print("\nTesting blockchain auditor...")
        
        from security.blockchain_audit import BlockchainAuditor
        
        # Initialize auditor (should use mock mode)
        auditor = BlockchainAuditor()
        
        # Test logging an event
        result = auditor.log_event("test_event", {"message": "System test"})
        print(f"✓ Event logged: {result}")
        
        # Test retrieving events
        events = auditor.get_events()
        print(f"✓ Retrieved {len(events)} events")
        
        print("✓ Blockchain auditor working in mock mode!")
        return True
        
    except Exception as e:
        print(f"❌ Blockchain auditor error: {e}")
        return False

if __name__ == "__main__":
    print("=== Federated Learning System Test ===")
    
    success = True
    success &= test_imports()
    success &= test_blockchain_mock()
    
    if success:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\nTo start your system:")
        print("1. Dashboard: streamlit run dashboard/app.py")
        print("2. Server: python server/federated_server.py") 
        print("3. Clients: python clients/blood_client.py (and others)")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")