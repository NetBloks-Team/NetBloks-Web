import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QTextEdit,
    QSizePolicy,
)
import os
import json
import PyQt6.QtCore as Qt
from google import genai
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QFont, QPen
from PyQt6.QtWidgets import QLabel
 
 
class GradientLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.text = text
        self.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
 
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
 
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
 
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, QColor("#6EBF7B"))
        gradient.setColorAt(1.0, QColor("#5CA4D6"))
 
        pen = QPen(gradient, 0)
        painter.setPen(pen)
 
        font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(font)
 
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text)
 
        painter.end()
 
 
def setGoal(goal):
    goal = goal[:400]
    with open("goal.txt", "w") as file:
        file.write(goal)
 
 
def getGoal():
    try:
        with open("goal.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""
 
 
def getApiKey():
    try:
        with open("api-key", "r") as file:
            api_key = file.read()
            return api_key
    except FileNotFoundError:
        print(
            "Please create a file called 'api-key' in the same directory as this script and paste your API key in it."
        )
        sys.exit(1)
 
 
def replace_ingredients(meal, ingredients):
    if meal and ingredients:
        genai.configure(api_key=getApiKey())
        model = genai.GenerativeModel("gemini-1.5-flash")
        goal = getGoal()
        response = model.generate_content(
            f"The goal is: {goal}. Replace the following ingredients with healthier alternatives for the meal according to the goal '{meal}': {json.dumps(ingredients)}. Provide the response as a python list of strings with each ingredient. It should be the same length as the input. If there is no good ingredient, insert an empty string."
        )._result
        response = str(response)[50:]
        end = response.find("}")
        response = (
            response[: end - 10]
            .replace("\\n", "")
            .replace("\\", "")
            .replace("python", "")
            .replace("```", "")
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error decoding the response.", response)
            return ["" for _ in ingredients]
    return []
 
 
from PyQt6.QtGui import QFont
 
 
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f7f9fc;
                color: #2c3e50;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            QLineEdit {
                background-color: #fff;
                border: 1px solid #dfe3e6;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            QFrame {
                border-radius: 15px;
                border: 1px solid #dfe3e6;
            }
            QTextEdit {
                border: 1px solid #dfe3e6;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """
        )
 
        self.layout = QVBoxLayout()

        self.header = GradientLabel("RecipeReboot")
        self.header.setStyleSheet("font-size: 24px, font-weight: bold")
        self.layout.addWidget(self.header)

        self.light_mode = True  # Start in light mode

        self.meal_name_layout = QHBoxLayout()
        self.meal_name_label = QLabel("Name of Meal: ")
        self.meal_name_label.setFont(QFont("Arial", 14))
        self.meal_name_input = QLineEdit()
        self.layout.addLayout(self.meal_name_layout)


        self.meal_name_layout.addWidget(self.meal_name_label)
        self.meal_name_layout.addWidget(self.meal_name_input)

        # Ingredients section
        self.ingredients_layout = QVBoxLayout()
        self.ingredients = []

        self.add_ingredient_button = QPushButton("Add Ingredient")
        self.add_ingredient_button.setFont(QFont("Arial", 14))
        self.add_ingredient_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            """
        )
        self.add_ingredient_button.clicked.connect(self.add_ingredient)
        self.layout.addLayout(self.ingredients_layout)
        self.layout.addWidget(self.add_ingredient_button)

        # Submit button
        self.submit_button = QPushButton("Load Alternative Ingredients")
        self.submit_button.setFont(QFont("Arial", 14))
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            """
        )
        self.submit_button.clicked.connect(self.submit)
        self.layout.addStretch(1)
        self.layout.addWidget(self.submit_button)

        # Horizontal layout for Set Goal and Toggle Theme buttons
        self.button_layout = QHBoxLayout()

        # Set Goal button
        self.set_goal_button = QPushButton("Set Goal")
        self.set_goal_button.setFont(QFont("Arial", 14))
        self.set_goal_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_goal_button.setStyleSheet(
            """
            QPushButton {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            }
            QPushButton:hover {
            background-color: #c0392b;
            }
            """
        )
        self.set_goal_button.clicked.connect(self.show_goal_dialog)
        self.button_layout.addWidget(self.set_goal_button)

        # Toggle theme button (with initial light mode)
        self.toggle_theme_button = QPushButton(
            "🌙", self
        )  # Set initial button text
        self.toggle_theme_button.setMinimumSize(self.toggle_theme_button.sizeHint())
        self.toggle_theme_button.setFont(QFont("Arial", 14))
        self.toggle_theme_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            """
        )
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        self.button_layout.addWidget(self.toggle_theme_button)

        # Add stretch to push the buttons to the right


        # Add the button layout to the main layout
        self.layout.addLayout(self.button_layout)

        # Set the main window background color to white
        self.setStyleSheet("background-color: #ecf0f1; color: black;")
        self.setLayout(self.layout)
 
        # Goal dialog frame
        self.goal_frame = QFrame(self)
        self.goal_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.goal_frame.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 0.95);
            border: 1px solid #bdc3e7;
            color: #333;
            border-radius: 15px;
        """
        )
        self.goal_frame.setGeometry(0, 0, self.width(), self.height())
        self.goal_frame.setVisible(False)
 
        goal_layout = QVBoxLayout()
        self.goal_input = QTextEdit(self.goal_frame)
        self.goal_input.setPlaceholderText("Enter your goal (max 400 characters)")
        self.goal_input.setFont(QFont("Arial", 14))
        self.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #dfe3e6;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )
        self.goal_input.setMaximumHeight(100)
        self.goal_input.textChanged.connect(self.limit_goal_text)
        goal_layout.addWidget(self.goal_input)
 
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self.goal_frame)
        ok_button.setFont(QFont("Arial", 14))
        ok_button.setStyleSheet(
            """
            QPushButton {
                background-color: #60a590;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
        """
        )
        cancel_button = QPushButton("Cancel", self.goal_frame)
        cancel_button.setFont(QFont("Arial", 14))
        cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
        """
        )
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        goal_layout.addLayout(button_layout)
 
        self.goal_frame.setLayout(goal_layout)
 
        ok_button.clicked.connect(self.set_goal)
        cancel_button.clicked.connect(self.hide_goal_dialog)

        # 2796-by-1290
 
        self.setGeometry(100, 100, 1290//4, 2796//4-50)
        self.apply_light_mode()
 
    def toggle_theme(self):
        if self.light_mode:
            self.apply_dark_mode()
            self.toggle_theme_button.setText(
                "☀️"
            )  # Change text to Light Mode
        else:
            self.apply_light_mode()
            self.toggle_theme_button.setText(
                "🌙"
            )  # Change text to Dark Mode
        self.light_mode = not self.light_mode  # Toggle mode
 
    def apply_light_mode(self):
        light_mode_style = """
            QWidget {
                background-color: #f7f9fc;
                color: #2c3e50;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            QLineEdit, QTextEdit {
                background-color: white;
                color: black;
                border: 1px solid #dfe3e6;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        self.setStyleSheet(light_mode_style)
        self.toggle_theme_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """
        )
        self.set_goal_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """
        )
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #599;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #488;
            }
            """
        )
 
    def apply_dark_mode(self):
        dark_mode_style = """
            QWidget {
            background-color: #2e2e2e;
            color: #f5f5f5;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            QLineEdit, QTextEdit {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 8px;
            }
            QPushButton {
            background-color: #555;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px;
            }
            QPushButton:hover {
            background-color: #444;
            }
            QTextEdit {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            }
        """
        self.toggle_theme_button.setStyleSheet(
            """
            QPushButton {
                background-color: #555;
            }
            QPushButton:hover {
                background-color: #444;
            }
            """
        )
        self.set_goal_button.setStyleSheet(
            """
            QPushButton {
                background-color: #555;
            }
            QPushButton:hover {
                background-color: #444;
            }
            """
        )
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #599;
            }
            QPushButton:hover {
                background-color: #488;
            }
            """
        )
        
        
        self.setStyleSheet(dark_mode_style)
 
    def show_goal_dialog(self):
        self.goal_frame.setGeometry(0, 0, self.width(), self.height())
        self.goal_frame.setVisible(True)
 
    def hide_goal_dialog(self):
        self.goal_frame.setVisible(False)
 
    def limit_goal_text(self):
        text = self.goal_input.toPlainText()
        if len(text) > 400:
            self.goal_input.setText(text[:400])
 
    def set_goal(self):
        goal = self.goal_input.toPlainText()
        if goal:
            setGoal(goal)
            print(f"Goal set to: {goal}")
        self.hide_goal_dialog()
 
    def add_ingredient(self):
        ingredient_layout = QHBoxLayout()
        ingredient_input = QLineEdit()
        ingredient_input.setFont(QFont("Arial", 14))
        ingredient_input.setStyleSheet(
            "padding: 5px; border: 1px solid #bdc3c7; border-radius: 5px; font-size: 14px;"
        )
        healthier_label = QLabel("")
        healthier_label.setFont(QFont("Arial", 9))
        remove_button = QPushButton("Remove")
        remove_button.setFont(QFont("Arial", 14))
        remove_button.setStyleSheet(
            "background-color: #944; color: white; padding: 5px; border: none; border-radius: 5px;"
        )
        remove_button.clicked.connect(lambda: self.remove_ingredient(ingredient_layout, healthier_label))
 
        ingredient_layout.addWidget(ingredient_input)
        ingredient_layout.addWidget(remove_button)
        self.ingredients_layout.addLayout(ingredient_layout)
        self.ingredients_layout.addWidget(healthier_label)
        self.ingredients.append((ingredient_layout, ingredient_input, healthier_label))
 
        # Ensure the goal frame is always on top
        self.goal_frame.raise_()
 
    def remove_ingredient(self, ingredient_layout, healthier):
        """
        Remove the ingredient's layout and widgets and adjust the overall layout.
        """
        # Loop through the ingredients layout to find the specific ingredient to remove
        for i in range(self.ingredients_layout.count()):
            layout_item = self.ingredients_layout.itemAt(i)
            if layout_item.layout() == ingredient_layout:
                # Remove all the widgets within the layout first
                for j in reversed(range(ingredient_layout.count())):
                    widget = ingredient_layout.itemAt(j).widget()
                    if widget:
                        widget.deleteLater()
                self.ingredients_layout.removeItem(layout_item)
                ingredient_layout.deleteLater()
                self.ingredients = [
                    ing for ing in self.ingredients if ing[0] != ingredient_layout
                ]
                # Remove the QLabel beneath the ingredient
                item = self.ingredients_layout.itemAt(i + 1)
                if item is not None:
                    healthier_label = item.widget()
                    if healthier_label:
                        healthier_label.deleteLater()
                break

        healthier.deleteLater()


        self.update()
 
    def submit(self):
        meal_name = self.meal_name_input.text()
        ingredients = [ing[1].text() for ing in self.ingredients]
 
        if meal_name and ingredients:
            healthier_ingredients = replace_ingredients(meal_name, ingredients)
 
            for i, (_, ingredient_input, healthier_label) in enumerate(
                self.ingredients
            ):
                healthier_label.setText(
                    f"Healthier alternative: {healthier_ingredients[i]}"
                )
        else:
            print("Meal name or ingredients are missing.")
 
    def setGeometry(self, x, y, width, height):
        self.move(x, y)
        self.resize(width, height)
        font_size = width // 20
        self.meal_name_label.setStyleSheet(f"font-size: {font_size}px;")
        self.meal_name_input.setStyleSheet(f"font-size: {font_size+10}px;")
        self.add_ingredient_button.setStyleSheet(f"font-size: {font_size}px;")
        self.submit_button.setStyleSheet(f"font-size: {font_size}px;")
        for ingredient_layout, ingredient_input, healthier_label in self.ingredients:
            ingredient_input.setStyleSheet(f"font-size: {font_size}px;")
            healthier_label.setStyleSheet(f"font-size: {font_size//2}px;")
        self.resize(width, height)
        self.meal_name_label.setStyleSheet(f"font-size: {width // 20}px;")
        self.meal_name_input.setStyleSheet(f"font-size: {width // 20}px;")
        self.add_ingredient_button.setStyleSheet(f"font-size: {width // 20}px;")
        self.submit_button.setStyleSheet(f"font-size: {width // 20}px;")
        for ingredient_layout, ingredient_input, healthier_label in self.ingredients:
            ingredient_input.setStyleSheet(f"font-size: {width // 25}px;")
            healthier_label.setStyleSheet(f"font-size: {width // 40}px;")
 
        self.goal_frame.setGeometry(0, 0, width, height)
        self.goal_input.setStyleSheet(
            f"font-size: {font_size}px; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px;"
        )
        for button in self.goal_frame.findChildren(QPushButton):
            if button.text() == "OK":
                button.setStyleSheet(
                    f"font-size: {font_size}px; padding: 10px; border: none; border-radius: 5px; background-color: #60a590; color: white;"
                )
            elif button.text() == "Cancel":
                button.setStyleSheet(
                    f"font-size: {font_size}px; padding: 10px; border: none; border-radius: 5px; background-color: #944; color: white;"
                )
 
        self.resize(width, height)
 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())