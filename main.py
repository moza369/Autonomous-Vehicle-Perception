#!/usr/bin/env python3
"""Autonomous Vehicle Perception System - Integrated Platform.

Launch the unified perception dashboard:
    python main.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    """Launch the Autonomous Vehicle Perception Dashboard."""
    import tkinter as tk
    from gui.dashboard import PerceptionDashboard
    
    root = tk.Tk()
    app = PerceptionDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
