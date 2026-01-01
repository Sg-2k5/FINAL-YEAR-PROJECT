"""
Dashboard launcher script.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from dashboard.admin_interface import launch_dashboard

if __name__ == "__main__":
    launch_dashboard()