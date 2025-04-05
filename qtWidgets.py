from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog, QLineEdit
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel, QFrame
from PyQt6.QtCore import Qt

class WorkspaceSelecter(QWidget): 
    def __init__(self, workspaces=None):
        super().__init__()
        # Create a dropdown (QComboBox)
        self.dropdown = QComboBox()
        if workspaces is not None:
            self.dropdown.addItems(workspaces)
        else:
            self.dropdown.addItems(["Workspace 1"])
        self.dropdown.currentIndexChanged.connect(self.on_workspace_change)

        # Create buttons for adding, deleting, and renaming workspaces
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_workspace)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_workspace)

        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_workspace)

        # Set up the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.dropdown)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.delete_button)
        self.button_layout.addWidget(self.rename_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)
    
    def setGeometry(self, x, y, width, height):
        super().setGeometry(x, y, width, height)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)


    def add_workspace(self):
        new_workspace_name = f"Workspace {self.dropdown.count() + 1}"
        self.dropdown.addItem(new_workspace_name)
        self.dropdown.setCurrentIndex(self.dropdown.count() - 1)

    def delete_workspace(self):
        current_index = self.dropdown.currentIndex()
        if current_index != -1:
            self.dropdown.removeItem(current_index)

    def rename_workspace(self):
        current_index = self.dropdown.currentIndex()
        if current_index != -1:
            new_name, ok = QInputDialog.getText(self, "Rename Workspace", "Enter new name:")
            if ok and new_name.strip():
                self.dropdown.setItemText(current_index, new_name.strip())

    def on_workspace_change(self, index):
        print(f"Selected Workspace: {self.dropdown.itemText(index)}")

class CodeEditor(QWidget):
    pass

class CodeBlock(QWidget):
    def __init__(self, type, parameters=[]):
        super().__init__()
        self.json_data = {
            "type": type,
            **{param: "" for param in parameters}
        }
        self.type = type
        self.parameters = parameters

        # Create a layout for the CodeBlock
        self.layout = QVBoxLayout()

        # Add a label at the top with the type

        self.type_label = QLabel(self.type)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.type_label)

        # Add a horizontal line below the label
        self.horizontal_line = QFrame()
        self.horizontal_line.setFrameShape(QFrame.Shape.HLine)
        self.horizontal_line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.horizontal_line)

        # Add text entry boxes for each parameter
        self.inputs = []
        for param in self.parameters:
            input_field = QWidget()
            input_layout = QHBoxLayout()
            input_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{param}:")
            text_entry = QLineEdit()
            text_entry.setPlaceholderText("Enter value here")  # Placeholder text
            text_entry.textChanged.connect(lambda value, param=param: self.on_value_change(param, value))  # Connect to handler
            self.inputs.append(text_entry)


            text_input = QLabel("Value")  # Placeholder for text input
            input_layout.addWidget(text_entry)

            input_field.setLayout(input_layout)
            self.layout.addWidget(input_field)

        # Set the layout for the widget
        self.setLayout(self.layout)

    def on_value_change(self, param, value):
        # Update the JSON data with the new value
        self.json_data[param] = value
        print(f"Updated {param} to {value}")