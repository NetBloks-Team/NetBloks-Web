import os
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QScrollArea, QHBoxLayout
import qtWidgets
import gemini_gen
from nn_wrapper import run_model
import threading
from PyQt6.QtCore import QMetaObject, Qt
from PyQt6.QtCore import Q_ARG

EPOCHS = 8

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None

        self.setWindowTitle("AI Project")
        # Set the geometry to the size of the screen
        screen_geometry = self.screen().availableGeometry()
        screen_geometry.setWidth(int(screen_geometry.width() * 0.75))
        screen_geometry.setHeight(int(screen_geometry.height() * 0.75))
        self.setGeometry(screen_geometry)

        
        # Create the workspace selector and code editor
        self.code_editor = qtWidgets.CodeEditor()
        self.workspace_selector = qtWidgets.WorkspaceSelecter(None, self.code_editor.get_savable, self.code_editor.from_savable, self.code_editor.clear)
        self.code_editor.set_saver(self.workspace_selector.save)
        # Create a layout and add the widgets
        layout = QVBoxLayout()
        layout.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignTop)
        # layout.addWidget(self.workspace_selector)  # Workspace selector at the top
        # Add a horizontal layout to consolidate the workspace selector in the left corner
        h_layout_left = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.on_run)

        self.cancel_button = QPushButton("Stop")
        self.cancel_button.clicked.connect(self.stop)
        self.cancel_button.hide()
        
        h_layout_left.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignLeft)
        h_layout_left.addWidget(self.workspace_selector)

        h_layout_right = QHBoxLayout()
        h_layout_right.setAlignment(qtWidgets.Qt.AlignmentFlag.AlignRight)
        h_layout_right.addWidget(self.run_button, alignment=qtWidgets.Qt.AlignmentFlag.AlignRight)
        h_layout_right.addWidget(self.cancel_button, alignment=qtWidgets.Qt.AlignmentFlag.AlignRight)
        h_layout = QHBoxLayout()
        h_layout.addLayout(h_layout_left)
        h_layout.addLayout(h_layout_right)
        layout.addLayout(h_layout)
        # Add a horizontal separator below the top bar
        separator = qtWidgets.QFrame()
        separator.setFrameShape(qtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(qtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        feedback_module = qtWidgets.FeedbackModule(gemini_gen.gemini_fb, gemini_gen.gemini_chatbot, self.code_editor.get_dataset, self.code_editor.get_json_data)
        feedback_module.setMaximumWidth(400)
        feedback_module.setMinimumWidth(400)

        mid_layout = QHBoxLayout()
        self.console = qtWidgets.Console()
        self.console.setMaximumWidth(400)
        mid_layout.addWidget(self.console)
        # Add a vertical separator between the console and the code editor
        vertical_separator = qtWidgets.QFrame()
        vertical_separator.setFrameShape(qtWidgets.QFrame.Shape.VLine)
        vertical_separator.setFrameShadow(qtWidgets.QFrame.Shadow.Sunken)
        mid_layout.addWidget(vertical_separator)
        mid_layout.addWidget(self.code_editor)
        vertical_separator2 = qtWidgets.QFrame()
        vertical_separator2.setFrameShape(qtWidgets.QFrame.Shape.VLine)
        vertical_separator2.setFrameShadow(qtWidgets.QFrame.Shadow.Sunken)
        mid_layout.addWidget(vertical_separator2)
        mid_layout.addWidget(feedback_module)
        # Add the code editor with scroll functionality
        layout.addLayout(mid_layout)
        
        # Set the layout to a container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_run(self):
        self.cancel_button.show()
        self.run_button.setEnabled(False)
        self.console.add_output("Initializing trainer...")

        # Get the code from the code editor
        code = self.code_editor.get_json_data()
        ds_name = self.code_editor.get_dataset()

        # Run the code in a separate thread
        def finish_run():
            gemini_gen.gemini_gen(ds_name, str(code))
            run_model(ds_name, self.console.add_output, nn_struct=str(code), epochs=EPOCHS)
            self.run_button.setEnabled(True)
            self.cancel_button.hide()

        # Create a new thread to run the code
        self.thread = threading.Thread(target=finish_run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.console.add_output("Stopping the training process...")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        
        self.cancel_button.hide()
        self.run_button.setEnabled(True)
        
