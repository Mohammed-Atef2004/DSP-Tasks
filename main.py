#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force matplotlib to use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from gui.app import DSPApplication

def main():
    root = tk.Tk()
    app = DSPApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()