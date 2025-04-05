"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog
from PyQt6.QtWidgets import QHBoxLayout
import application
from gemini_gen import gemini_gen

#-----Function Section-----

def main():
    """
    docstring thing goes here
    """
    with open("llm_output.py", "w", encoding="utf-8") as f:
        f.write(gemini_gen())
    os.system("python llm_output.py")
        
        

def qt_main():
    app = QApplication([])
    window = application.MainWindow()
    window.show()
    app.exec()    

if __name__ == "__main__":
    qt_main()
