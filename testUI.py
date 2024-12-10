import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout,
                             QMessageBox, QDialog, QTextEdit)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ввод и вывод текста")
        self.resize(600, 300)

        self.input_label = QLabel("Введите текст:")
        self.input_line = QLineEdit()
        self.submit_button = QPushButton("Отправить")
        self.submit_button.clicked.connect(self.send_text)
        self.help_button = QPushButton("Справка")
        self.help_button.clicked.connect(self.show_help)
        self.output_label = QLabel("Выводимый текст:")
        self.output_text = QTextEdit()
        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_line)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.help_button)  #
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_text)
        self.setLayout(layout)

    def send_text(self):
        text = self.input_line.text()
        if text:
            self.output_text.setText(text)
        else:
            QMessageBox.warning(self, "Ошибка", "Введите текст!")

    def show_help(self):
        help_dialog = HelpDialog()
        help_dialog.exec()


class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Справка")
        self.help_text = QLabel()
        self.help_text.setText("Справка по работе с приложением")

        layout = QVBoxLayout()
        layout.addWidget(self.help_text)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec())
