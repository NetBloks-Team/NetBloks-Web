import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QInputDialog, QLineEdit, QScrollArea, QHBoxLayout, QSizePolicy
from PyQt6.QtWidgets import QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPainter, QColor
import os
import threading

LAYER_BLOCK_COLOR = QColor(70, 90, 126)      # Blue-gray for layers
ACTIVATION_BLOCK_COLOR = QColor(126, 87, 107)

datasets = ["MNIST", "CIFAR 10", "CIFAR 100", "KMNIST", "SVHN"]

activations = {
    "ELU": ["alpha"],
    "LeakyReLU": ["negative_slope"],
    "LogSoftmax": [],
    "PReLU": ["num_parameters"],
    "ReLU": [],
    "SELU": [],
    "Sigmoid": [],
    "Softmax": [],
    "Softplus": ["beta"],
    "Softshrink": ["lambd"],
    "Tanh": [],
    "Threshold": ["threshold", "value"],
}

layers = {
    "AdaptiveAvgPool2d": ["output_size"],
    "AdaptiveMaxPool2d": ["output_size"],
    "BatchNorm2d": ["num_features"],
    "Conv2d": ["out_channels", "kernel_size", "stride"],
    "Conv3d": ["out_channels", "kernel_size", "stride", "padding"],
    "ConvTranspose2d": ["out_channels", "kernel_size", "stride", "padding"],
    "ConvTranspose3d": ["out_channels", "kernel_size", "stride", "padding"],
    "Dropout": ["p"],
    "Flatten": [],
    "GRU": ["hidden_size", "num_layers", "bidirectional"],
    "Linear": ["out_features"],
    "LSTM": ["hidden_size", "num_layers", "bidirectional"],
    "MaxPool2d": ["kernel_size", "stride"],
    "PixelShuffle": ["upscale_factor"],
    "PixelUnshuffle": ["downscale_factor"],
    "Transformer": ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers"],
    "TransformerDecoder": ["decoder_layer", "num_layers"],
    "TransformerDecoderLayer": ["d_model", "nhead", "dim_feedforward"],
    "TransformerEncoder": ["encoder_layer", "num_layers"],
    "TransformerEncoderLayer": ["d_model", "nhead", "dim_feedforward"],
    "Upsample": ["size", "scale_factor", "mode"],
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
            # Save the data to a file or database
            if not name:
                name = self.dropdown.currentText()
            
            os.makedirs("saves", exist_ok=True)
            with open(f"saves/{name}.json", "w") as f:
                json.dump(data, f)

    def on_workspace_change(self, index):
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
        self.layout.setContentsMargins(10, 0, 10, 0)
        
        # Create a menu for adding and deleting code blocks
        self.menu_layout = QHBoxLayout()
        self.add_block_button = QPushButton("Add Layer Block")
        self.add_block_button.clicked.connect(self.add_code_block)
        self.add_activation_button = QPushButton("Add Activation Block")
        self.add_activation_button.clicked.connect(self.add_activation)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        
        # Add a label for the dataset dropdown with a fixed size
        self.dataset_label = QLabel("Dataset:")
        self.dataset_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.menu_layout.addWidget(self.dataset_label)

        # Create a dropdown for selecting datasets
        self.dataset_dropdown = QComboBox()
        self.dataset_dropdown.addItems(datasets)
        self.menu_layout.addWidget(self.dataset_dropdown)

        self.menu_layout.addWidget(self.add_block_button)
        self.menu_layout.addWidget(self.add_activation_button)
        self.menu_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.menu_layout)

        horizontal_line = QFrame()
        horizontal_line.setFrameShape(QFrame.Shape.HLine)
        horizontal_line.setStyleSheet("background-color: #333333;")
        self.layout.addWidget(horizontal_line)

        # Create a scroll area for the code blocks
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Enable resizing of the scroll area content

        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)
        self.scroll_area_layout.setSpacing(10)  # Add spacing between blocks
        self.scroll_area_layout.setContentsMargins(5, 0, 5, 0)
        self.scroll_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Stack blocks at the top

        self.scroll_area.setWidget(self.scroll_area_content)
        
        # Set fixed width for scroll area only
        self.scroll_area.setFixedWidth(600)
        
        # Create a container widget for the scroll area
        scroll_container = QWidget()
        scroll_container_layout = QHBoxLayout(scroll_container)
        scroll_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll_container_layout.addWidget(self.scroll_area)
        scroll_container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add the scroll container to the main layout
        self.layout.addWidget(scroll_container)
        
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
        
    def set_explainer(self, explain_layer):
        # Set the explain_layer function for each block
        for i in range(self.blocks_container.count()):
            block = self.blocks_container.itemAt(i).widget()
            if block:
                block.explain_layer = explain_layer

    def get_json_data(self):
        # Collect JSON data from all code blocks
        json_data = []
        for i in range(self.blocks_container.count()):
            block = self.blocks_container.itemAt(i).widget()
            if block:
                try:
                    json_data.append(block.json_data)
                except Exception as e:
                    pass
        return json_data

    def get_dataset(self):
        return self.dataset_dropdown.currentText()

    def get_explain_buttons(self):
        # Get the explain buttons from all code blocks
        explain_buttons = []
        for i in range(self.blocks_container.count()):
            block = self.blocks_container.itemAt(i).widget()
            if block:
                explain_buttons.append(block.explain_button)
        return explain_buttons
    
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
        self.explain_layer = None
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

        self.explain_button = QPushButton("?")
        self.explain_button.setFixedSize(30, 30)
        self.explain_button.clicked.connect(lambda: self.explain_layer(self.json_data) if self.explain_layer else None)

        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.type_label = QLabel(self.type)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = self.type_label.font()
        font.setPointSize(12)  # Increase the font size
        self.type_label.setFont(font)

        delete_button = QPushButton("X")
        delete_button.setFixedSize(30, 30)
        delete_button.clicked.connect(self.deleteLater)

        top_bar_layout.addWidget(self.type_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.explain_button)
        top_bar_layout.addWidget(delete_button)

        top_bar.setLayout(top_bar_layout)
        self.layout.addWidget(top_bar)
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if parameters:
            # Add a horizontal line below the label
            self.horizontal_line = QFrame()
            self.horizontal_line.setFrameShape(QFrame.Shape.HLine)
            self.horizontal_line.setStyleSheet("background-color: black; border: none;")
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
    
    def set_saver(self, save):
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
        if self.save:
            self.save()
    
    def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self.type in layers.keys():
                painter.setBrush(LAYER_BLOCK_COLOR)
            else:
                painter.setBrush(ACTIVATION_BLOCK_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.rect(), 10, 10)

class Console(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)
        self.setWindowTitle("Console")
        title_label = QLabel("Console Output")
        title_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        font = title_label.font()
        font.setPointSize(16)
        title_label.setFont(font)
        self.layout.addWidget(title_label, stretch=0)
        self.console_output = QScrollArea()
        self.console_output.setWidgetResizable(True)
        self.console_content = QLabel()
        self.console_content.setWordWrap(True)
        self.console_content.setStyleSheet("background-color: #444444; padding: 10px; border-radius: 5px;")
        self.console_content.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.console_output.setWidget(self.console_content)
        self.layout.addWidget(self.console_output, stretch=1)
        self.add_output("Console output will appear here. For more information, check the terminal.")
        self.console_output.setAlignment(Qt.AlignmentFlag.AlignTop)


    @pyqtSlot(str)
    def add_output(self, text):
        self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum())
        max_length = 40
        text = text.split(" ")
        wrapped_text = ""
        for word in text:
            if len(wrapped_text.split("\n")[-1]) + len(word) + 1 > max_length:
                if len(word) > max_length:
                    wrapped_text += "\n" + word[:max_length-1] + "-\n" + word[max_length-1:]
                wrapped_text += "\n" + word
            else:
                wrapped_text += " " + word

        text = wrapped_text.strip()
        self.console_content.setText(self.console_content.text() + "\n\n" + text + "\n")
        self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum())

class FeedbackModule(QWidget):
    def __init__(self, general_feedback=None, chatbot=None, getting_started=None, dataset_getter=None, code_getter=None, explain_layer=None, explain_buttons=None):
        super().__init__()
        self.chatbot = chatbot
        self.general_feedback = general_feedback
        self.dataset_getter = dataset_getter
        self.code_getter = code_getter
        self.explainer = explain_layer
        self.explain_buttons = explain_buttons
        self.starter = getting_started

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)

        # Chatbot response box
        self.chatbot_scroll_area = QScrollArea()
        self.chatbot_scroll_area.setWidgetResizable(True)
        self.chatbot_output = QLabel()
        self.chatbot_output.setWordWrap(True)
        self.chatbot_output.setStyleSheet("background-color:rgb(70, 70, 70); padding: 10px; border-radius: 5px;")
        self.chatbot_output.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chatbot_output.setText("Chatbot responses will appear here.")
        self.chatbot_scroll_area.setWidget(self.chatbot_output)
        self.layout.addWidget(self.chatbot_scroll_area, stretch=1)

        # Input field and send button
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        self.layout.addLayout(input_layout)

        # General feedback button
        self.getting_started_button = QPushButton("Getting Started")
        self.getting_started_button.clicked.connect(self.getting_started)
        self.layout.addWidget(self.getting_started_button)

        self.feedback_button = QPushButton("Provide Feedback on Model")
        self.feedback_button.clicked.connect(self.provide_feedback)
        self.layout.addWidget(self.feedback_button)

    @pyqtSlot()
    def send_message(self):
        message = self.message_input.text().strip()
        if message:
            self.chatbot_output.setText(self.chatbot_output.text() + f"\n\nYou: {message}\n\n")
            self.message_input.clear()
            response = None
            if self.chatbot:
                self.message_input.setDisabled(True)
                self.send_button.setDisabled(True)
                self.feedback_button.setDisabled(True)
                self.getting_started_button.setDisabled(True)
                
                for button in self.explain_buttons():
                    button.setDisabled(True)
                def finish():
                    response = self.chatbot(self.dataset_getter(), self.code_getter(), message)
                    self.chatbot_output.setText(self.chatbot_output.text() + f"Chatbot: {response}")
                    self.message_input.setDisabled(False)
                    self.send_button.setDisabled(False)
                    self.feedback_button.setDisabled(False)
                    self.getting_started_button.setDisabled(False)
                    for button in self.explain_buttons():
                        button.setDisabled(False)
                thread = threading.Thread(target=finish, daemon=True)
                thread.start()
            else:
                response = "No chatbot function provided."
                self.chatbot_output.setText(self.chatbot_output.text() + f"Chatbot: {response}")

    @pyqtSlot()
    def provide_feedback(self):
        self.chatbot_output.setText(self.chatbot_output.text() + "\n\nRequesting feedback from the model...\n\n")
        response = None
        if self.general_feedback:
            self.message_input.setDisabled(True)
            self.send_button.setDisabled(True)
            self.feedback_button.setDisabled(True)
            self.getting_started_button.setDisabled(True)

            for button in self.explain_buttons():
                    button.setDisabled(True)
            def finish():
                response = self.general_feedback(self.dataset_getter(), self.code_getter())
                self.chatbot_output.setText(self.chatbot_output.text() + f"Chatbot: {response}")
                self.message_input.setDisabled(False)
                self.send_button.setDisabled(False)
                self.feedback_button.setDisabled(False)
                self.getting_started_button.setDisabled(False)
                for button in self.explain_buttons():
                    button.setDisabled(False)
            thread = threading.Thread(target=finish, daemon=True)
            thread.start()
        else:
            response = "No feedback function provided."
            self.chatbot_output.setText(self.chatbot_output.text() + "Feedback:\n"+response)
    
    @pyqtSlot()
    def getting_started(self):
        self.chatbot_output.setText(self.chatbot_output.text() + "\n\nRequesting Starting Guide...\n\n")
        response = None
        if self.general_feedback:
            self.message_input.setDisabled(True)
            self.send_button.setDisabled(True)
            self.feedback_button.setDisabled(True)
            self.getting_started_button.setDisabled(True)

            for button in self.explain_buttons():
                    button.setDisabled(True)
            def finish():
                response = self.starter(self.dataset_getter(), layers, activations)
                self.chatbot_output.setText(self.chatbot_output.text() + f"Chatbot: {response}")
                self.message_input.setDisabled(False)
                self.send_button.setDisabled(False)
                self.feedback_button.setDisabled(False)
                self.getting_started_button.setDisabled(False)
                for button in self.explain_buttons():
                    button.setDisabled(False)
            thread = threading.Thread(target=finish, daemon=True)
            thread.start()
        else:
            response = "No feedback function provided."
            self.chatbot_output.setText(self.chatbot_output.text() + response)

    # In the FeedbackModule class, update the explain_layer method:

    @pyqtSlot(dict)
    def explain_layer(self, layer_data):
        """Explain a specific layer"""
        self.chatbot_output.setText(self.chatbot_output.text() + "\n\nExplaining layer...\n\n")
        
        if self.explain_layer:
            self.message_input.setDisabled(True)
            self.send_button.setDisabled(True)
            self.feedback_button.setDisabled(True)
            self.getting_started_button.setDisabled(True)
            for button in self.explain_buttons():
                    button.setDisabled(True)
            
            def finish():
                # Get dataset and code data first
                dataset = self.dataset_getter()
                nn_struct = self.code_getter()
                # Now call the explain_layer function with all required arguments
                response = self.explainer(dataset, nn_struct, layer_data)
                self.chatbot_output.setText(self.chatbot_output.text() + f"Explanation: {response}")
                
                for button in self.explain_buttons():
                    button.setDisabled(False)
                self.message_input.setDisabled(False)
                self.send_button.setDisabled(False)
                self.feedback_button.setDisabled(False)
                self.getting_started_button.setDisabled(False)
                
            thread = threading.Thread(target=finish, daemon=True)
            thread.start()
        else:
            response = "No explanation function provided."
            self.chatbot_output.setText(self.chatbot_output.text() + "Explanation:\n" + response)
