#!/usr/bin/env python3
"""
意识工程化构造实验 v4.6 - 启动脚本
Consciousness Engineering Construction Experiment v4.6 - Launcher

Usage:
    python run.py
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consciousness_sim.main import main

if __name__ == "__main__":
    main()
