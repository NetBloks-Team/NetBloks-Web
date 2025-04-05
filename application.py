import os
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QScrollArea, QHBoxLayout
import qtWidgets
import gemini_gen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Project")
        # Set the geometry to the size of the screen
        screen_geometry = self.screen().geometry()
        self.setGeometry(screen_geometry)
        
        # Create the workspace selector and code editor
        self.code_editor = qtWidgets.CodeEditor()
        self.workspace_selector = qtWidgets.WorkspaceSelecter(None, self.code_editor.get_savable, self.code_editor.from_savable, self.code_editor.clear)
        
        # Create a layout and add the widgets
        layout = QVBoxLayout()
        layout.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignTop)
        # layout.addWidget(self.workspace_selector)  # Workspace selector at the top
        # Add a horizontal layout to consolidate the workspace selector in the left corner
        h_layout_left = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.on_run)
        
        h_layout_left.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignLeft)
        h_layout_left.addWidget(self.workspace_selector)

        h_layout_right = QHBoxLayout()
        h_layout_right.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignRight)
        h_layout_right.addWidget(self.run_button, alignment=qtWidgets.Qt.AlignmentFlag.AlignRight)
        h_layout = QHBoxLayout()
        h_layout.addLayout(h_layout_left)
        h_layout.addLayout(h_layout_right)
        layout.addLayout(h_layout)
        # Add the code editor with scroll functionality
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.code_editor)
        layout.addWidget(scroll_area)
        
        # Set the layout to a container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_run(self):
        # Get the selected workspace
        
        # Get the code from the code editor
        code = self.code_editor.get_json_data()
        dataset = self.code_editor.get_dataset()
        
        # Run the code in the selected workspace
        gemini_gen.gemini_gen(dataset, code)
        os.system(f"python3 nn_wrapper.py")