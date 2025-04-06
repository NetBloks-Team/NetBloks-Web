"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import qtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import application
     

def main():
    app = QApplication([])
    window = application.MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()