import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog, QLineEdit, QScrollArea, QHBoxLayout
from PyQt6.QtWidgets import QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
import os

datasets = ["MNIST", "CIFAR 10", "CIFAR 100", "IMDB"]

layers = {
    "Conv2d": ["channels","kernel_size", "stride"],
    "Linear": ["out_features"],
    "MaxPool2d": ["kernel_size", "stride"],
    "Dropout": ["p"],
    "BatchNorm2d": ["num_features"],
    "Flatten": [],
    "AdaptiveAvgPool2d": ["output_size"],
    "AdaptiveMaxPool2d": ["output_size"],
    "Upsample": ["size", "scale_factor", "mode"],
    "PixelShuffle": ["upscale_factor"],
    "PixelUnshuffle": ["downscale_factor"],
    "ConvTranspose2d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "Conv3d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "ConvTranspose3d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "LSTM": ["input_size", "hidden_size", "num_layers", "bidirectional"],
    "GRU": ["input_size", "hidden_size", "num_layers", "bidirectional"],
    "RNN": ["input_size", "hidden_size", "num_layers", "bidirectional"],
    "Transformer": ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers"],
    "TransformerEncoder": ["encoder_layer", "num_layers"],
    "TransformerDecoder": ["decoder_layer", "num_layers"],
    "TransformerEncoderLayer": ["d_model", "nhead", "dim_feedforward"],
    "TransformerDecoderLayer": ["d_model", "nhead", "dim_feedforward"],
}

activations = {
    "ReLU": [],
    "Sigmoid": [],
    "Tanh": [],
    "Softmax": [],
    "LogSoftmax": [],
    "LeakyReLU": ["negative_slope"],
    "ELU": ["alpha"],
    "SELU": [],
    "PReLU": ["num_parameters"],
    "Softplus": ["beta"],
    "Softshrink": ["lambd"],
    "Threshold": ["threshold", "value"],
}

class WorkspaceSelecter(QWidget): 
    def __init__(self, workspaces=None, get_savable=None, from_savable=None, clear=None):
        super().__init__()
        self.get_savable = get_savable
        self.from_savable = from_savable
        self.clear_editor = clear
        self.previous_workspace = None

        # Create a dropdown (QComboBox)
        self.dropdown = QComboBox()
        if workspaces is not None:
            self.previous_workspace = workspaces[0]
            self.dropdown.addItems(workspaces)
        elif self.get_file_paths():
            self.previous_workspace = self.get_file_paths()[0].strip(".json")
            self.dropdown.addItems([file.strip(".json") for file in self.get_file_paths()])
            with open(f"saves/{self.previous_workspace}.json", "r") as f:
                data = json.loads(f.read())
                self.from_savable(data)
        else:
            self.previous_workspace = "Workspace 1"
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
        self.layout = QHBoxLayout()
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

    def get_file_paths(self):
        # Get the file paths of all workspaces
        if not os.path.exists("saves"):
            return []
        return [f for f in os.listdir("saves") if os.path.isfile(os.path.join("saves", f))]


    def add_workspace(self):
        self.clear_editor()
        new_workspace_name = f"Workspace {self.dropdown.count() + 1}"
        while new_workspace_name in [self.dropdown.itemText(i) for i in range(self.dropdown.count())]:
            new_workspace_name = f"Workspace {self.dropdown.count() + 2}"
        self.dropdown.addItem(new_workspace_name)
        self.dropdown.setCurrentIndex(self.dropdown.count() - 1)
        self.on_workspace_change(self.dropdown.count() - 1)

    def delete_workspace(self):
        current_index = self.dropdown.currentIndex()
        file_path = f"saves/{self.dropdown.itemText(current_index)}.json"
        if current_index != -1:
            self.dropdown.removeItem(current_index)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    def rename_workspace(self):
        current_index = self.dropdown.currentIndex()
        if current_index != -1:
            new_name, ok = QInputDialog.getText(self, "Rename Workspace", "Enter new name:")
            if ok and new_name.strip():
                self.dropdown.setItemText(current_index, new_name.strip())
    
    def set_saves(self, get_savable, from_savable, clear):
        self.get_savable = get_savable
        self.from_savable = from_savable
        self.clear_editor = clear
    
    def save(self, name=None):
        if self.get_savable is not None:
            data = self.get_savable()
            print(data)
            # Save the data to a file or database
            if not name:
                name = self.dropdown.currentText()
            
            os.makedirs("saves", exist_ok=True)
            with open(f"saves/{name}.json", "w") as f:
                json.dump(data, f)

    def on_workspace_change(self, index):
        print(f"Workspace changed to: {self.dropdown.itemText(index)}")
        if self.previous_workspace in [self.dropdown.itemText(i) for i in range(self.dropdown.count())]:
            self.save(self.previous_workspace)
        if self.from_savable is not None:
            # Load the data from a file or database
            name = self.dropdown.itemText(index)
            try:
                with open(f"saves/{name}.json", "r") as f:
                    data = json.load(f)
                self.from_savable(data)
            except FileNotFoundError:
                self.clear_editor()
        self.previous_workspace = self.dropdown.currentText()
            
        
class CodeEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.save = None
        self.layout = QVBoxLayout()
        # Create a menu for adding and deleting code blocks
        self.menu_layout = QHBoxLayout()
        self.add_block_button = QPushButton("Add Layer Block")
        self.add_block_button.clicked.connect(self.add_code_block)
        self.add_activation_button = QPushButton("Add Activation Block")
        self.add_activation_button.clicked.connect(self.add_activation)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        # Create a dropdown for selecting datasets
        self.dataset_dropdown = QComboBox()
        self.dataset_dropdown.addItems(datasets)
        self.menu_layout.addWidget(self.dataset_dropdown)

        self.print_button = QPushButton("Print JSON")
        self.print_button.clicked.connect(lambda: print(self.get_json_data()))

        self.menu_layout.addWidget(self.add_block_button)
        self.menu_layout.addWidget(self.add_activation_button)
        self.menu_layout.addWidget(self.print_button)
        self.menu_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.menu_layout)
        
        # Create a scroll area for the code blocks
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Enable resizing of the scroll area content

        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)
        self.scroll_area_layout.setSpacing(10)  # Add spacing between blocks
        self.scroll_area_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Stack blocks at the top

        self.scroll_area.setWidget(self.scroll_area_content)

        self.scroll_area_container = QVBoxLayout()
        self.scroll_area_container.setAlignment(Qt.AlignmentFlag.AlignTop)  # Align scroll area to the top
        self.scroll_area_container.addWidget(self.scroll_area)
        self.layout.addLayout(self.scroll_area_container)

        # Initialize blocks_container
        self.blocks_container = self.scroll_area_layout

        self.setLayout(self.layout)     

    def set_saver(self, save):
        self.save = save
        for i in range(self.blocks_container.count()):
            block = self.blocks_container.itemAt(i).widget()
            if block:
                block.set_saver(save)

    def add_code_block(self):
        # Prompt the user to select a block type
        block_type, ok = QInputDialog.getItem(
            self, "Select Code Block Type", "Block Type:", layers.keys(), 0, False
        )
        if ok and block_type:
            # Create a new CodeBlock and add it to the container
            parameters = layers[block_type]
            new_block = CodeBlock(block_type, parameters)
            self.blocks_container.addWidget(new_block)
        
        if self.save:
            self.save()

    def add_activation(self):
        block_type, ok = QInputDialog.getItem(
            self, "Select Code Block Type", "Block Type:", activations.keys(), 0, False
        )
        if ok and block_type:
            # Create a new CodeBlock and add it to the container
            parameters = activations[block_type]
            new_block = CodeBlock(block_type, parameters)
            self.blocks_container.addWidget(new_block)
        
        if self.save:
            self.save()

    def get_json_data(self):
        # Collect JSON data from all code blocks
        json_data = []
        for i in range(self.blocks_container.count()):
            block = self.blocks_container.itemAt(i).widget()
            if block:
                json_data.append(block.json_data)
        return json_data

    def get_dataset(self):
        return self.dataset_dropdown.currentText()
    
    def get_savable(self):
        # Get the JSON data from all code blocks
        json_data = self.get_json_data()
        # Get the selected dataset
        dataset = self.dataset_dropdown.currentText()
        # Create a dictionary to hold the data
        data = {
            "dataset": dataset,
            "blocks": json_data
        }
        return data
    
    def from_savable(self, data):
        self.clear()
        # Set the dataset
        self.dataset_dropdown.setCurrentText(data["dataset"])
        # Clear existing blocks
        for i in reversed(range(self.blocks_container.count())):
            widget = self.blocks_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        # Add blocks from the data
        for block_data in data["blocks"]:
            block_type = block_data["type"]
            parameters = layers.get(block_type, [])
            new_block = CodeBlock(block_type, parameters, self.save)
            new_block.setValues(block_data)
            self.blocks_container.addWidget(new_block)
    
    def clear(self):
        # Clear the code editor
        for i in reversed(range(self.blocks_container.count())):
            widget = self.blocks_container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.dataset_dropdown.setCurrentIndex(0)

        if self.save:
            self.save()

class CodeBlock(QWidget):
    def __init__(self, type, parameters=[], save=None):
        super().__init__()
        self.save = save
        self.json_data = {
            "type": type,
            **{param: "" for param in parameters}
        }
        self.type = type
        self.parameters = parameters
        # Set the background color to grey
        # Create a layout for the CodeBlock
        # Set the background color and border for the CodeBlock using a QFrame
        self.setAutoFillBackground(True)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)  # Add padding inside the box

        # Add a label at the top with the type
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.type_label = QLabel(self.type)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = self.type_label.font()
        font.setPointSize(12)  # Increase the font size
        self.type_label.setFont(font)

        delete_button = QPushButton("X")
        delete_button.setFixedSize(20, 20)
        delete_button.clicked.connect(self.deleteLater)

        top_bar_layout.addWidget(self.type_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(delete_button)

        top_bar.setLayout(top_bar_layout)
        self.layout.addWidget(top_bar)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.type_label)

        if parameters:
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

            input_layout.addWidget(label)
            input_layout.addWidget(text_entry)

            input_field.setLayout(input_layout)
            self.layout.addWidget(input_field)

        # Set the layout for the widget
        self.setLayout(self.layout)
    
    def set_save(self, save):
        self.save = save

    def setValues(self, values):
        # Set the values for each parameter
        self.json_data.update(values)
        for i, param in enumerate(self.parameters):
            if i < len(values):
                self.inputs[i].setText(str(values[param]))

    def on_value_change(self, param, value):
        # Update the JSON data with the new value
        self.json_data[param] = value
        print(f"Updated {param} to {value}")
        if self.save:
            self.save()
    
    def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(Qt.GlobalColor.darkGray)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.rect(), 10, 10)

class Console(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("Console")
        title_label = QLabel("Console Output")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        title_label.setFont(font)
        self.layout.addWidget(title_label)
        self.console_output = QLabel()


    def add_output(self, text):
        self.console_output.setText(self.console_output.text() + "\n" + text)
        self.layout.addWidget(self.console_output)