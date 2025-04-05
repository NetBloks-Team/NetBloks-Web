"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13
"""

#-----Import Section-----

import os
import qtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog
from PyQt6.QtWidgets import QHBoxLayout
import application
from gemini_gen import gemini_gen

#-----Function Section-----

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workspace Selector")
        
        # Create a combo box as a placeholder for qtWidgets.WorkspaceSelecter
        self.test_widget = qtWidgets.CodeEditor() #CodeBlock("Conv2d", ["kernel", "Stride", "Padding"])
        self.test_widget.setGeometry(0, 0, 200, 100)
        # self.workspace_selector.addItems(["Workspace 1", "Workspace 2", "Workspace 3"])
        
        # Set up the central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.test_widget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        

def main():
    app = QApplication([])
    window = application.MainWindow()
    window.show()
    app.exec()    

if __name__ == "__main__":
    main()