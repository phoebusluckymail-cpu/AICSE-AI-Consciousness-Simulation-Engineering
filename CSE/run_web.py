#!/usr/bin/env python3
"""启动 Web UI"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.system("streamlit run consciousness_sim/app.py --server.port 8501")
