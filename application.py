from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog
from PyQt6.QtWidgets import QHBoxLayout
import qtWidgets

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Project")
        
        # Create the workspace selector and code editor
        self.workspace_selector = qtWidgets.WorkspaceSelecter()
        self.code_editor = qtWidgets.CodeEditor()
        
        # Create a layout and add the widgets
        layout = QVBoxLayout()
        layout.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.workspace_selector)  # Workspace selector at the top
        layout.addWidget(self.code_editor)        # Code editor below
        
        # Set the layout to a container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)