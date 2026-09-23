"""
Pytest configuration ensuring root directory is included in Python path.
"""
import sys
import os

# Add root directory to sys.path for test resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
