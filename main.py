from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog
from PyQt6.QtWidgets import QHBoxLayout
import qtWidgets

"""
An AI project for the UNR ACM 2025 Hackathon

Authors: Max Clemetsen and Kevin Pettibone
Python version: 3.13.2
"""

#-----Import Section-----

import os
from google import genai

#-----Function Section-----

def main():
    """
    docstring thing goes here
    """
    print("You are using API key:", os.environ["GEMINI_KEY"])
    #Make sure to put your Gemini API key in the environment variables
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    with open("nn.json", "r", encoding="utf-8") as f:
        prompt = f.read()
    full_prompt = "Please generate python code for a pytorch neural network that has the following parameters:\n" + prompt +"\nOnly give code, and do not reply with anything else. Give the code as an entire file that can be executed."
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21", contents=full_prompt
    )
    output = response.text.strip("```").removeprefix("python\n")
    print(output)
    with open("llm_output.py", "w", encoding="utf-8") as f:
        f.write(output)
    os.system("python llm_output.py")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workspace Selector")
        
        # Create a combo box as a placeholder for qtWidgets.WorkspaceSelecter
        self.test_widget = qtWidgets.WorkspaceSelecter() #CodeBlock("Conv2d", ["kernel", "Stride", "Padding"])
        self.test_widget.setGeometry(0, 0, 200, 100)
        # self.workspace_selector.addItems(["Workspace 1", "Workspace 2", "Workspace 3"])
        
        # Set up the central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.test_widget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        

def qt_main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()    

if __name__ == "__main__":
    main()
