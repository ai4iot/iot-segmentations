import os
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, \
    QGridLayout, QFileDialog, QMessageBox
from ..models import ModelBuilder

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900, 600)

        self.setWindowTitle("Main Menu")
        self.setWindowIcon(QIcon('icon.png'))

        self.model_name = 'efficientnet_b0'
        self.mode = 'train'
        self.weights_path = './ai_operations/models/person_nonperson_efficientnet.pth'
        self.dataset_path = '../../input/test/esp-camera'
        self.output_path = './predictions'
        self.image_input_path = '0'  # Default value for QLineEdit
        self.epochs = 10
        self.learning_rate = '0.0001'
        self.save_images_checked = False
        self.visualize_checked = False
        self.pretrained_checked = True
        self.fine_tune_checked = True

        # Scroll menu for Model Name
        model_name_label = QLabel("Model Name:")
        self.model_name_combo = QComboBox()
        model_name_values = ["efficientnet_b0", "resnet18", "Model C"]
        self.model_name_combo.addItems(model_name_values)

        # Scroll menu for Mode
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        mode_values = ["Train", "Local Inference", "Live Inference", "Metrics"]
        self.mode_combo.addItems(mode_values)
        self.mode_combo.currentIndexChanged.connect(self.handle_mode_change)

        # Text input for Weights
        weights_label = QLabel("Weights:")
        self.weights_entry = QLineEdit(self.weights_path)
        self.weights_button = QPushButton("Select Directory")
        self.weights_button.clicked.connect(self.open_weights_dialog)

        # Text input for Dataset
        dataset_label = QLabel("Dataset:")
        self.dataset_entry = QLineEdit(self.dataset_path)
        self.dataset_button = QPushButton("Select Directory")
        self.dataset_button.clicked.connect(self.open_dataset_dialog)

        # Text output for Output
        output_label = QLabel("Output:")
        self.output_entry = QLineEdit(self.output_path)
        self.output_button = QPushButton("Select Directory")
        self.output_button.clicked.connect(self.open_output_dialog)

        # Radio button for Save Images
        save_images_label = QLabel("Save Images:")
        self.save_images_checkbox = QCheckBox()

        # Text input for Image Input
        image_input_label = QLabel("Video input for live inference:")
        self.image_input_entry = QLineEdit(self.image_input_path)

        # Radio button for Visualize
        visualize_label = QLabel("Visualize:")
        self.visualize_checkbox = QCheckBox()

        # Text input for Epochs
        epochs_label = QLabel("Epochs:")
        self.epochs_entry = QLineEdit(str(self.epochs))  # Convert to string

        # Radio button for Pretrained
        pretrained_label = QLabel("Pretrained:")
        self.pretrained_checkbox = QCheckBox()

        # Text input for Learning Rate
        learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_entry = QLineEdit(self.learning_rate)

        # Radio button for Fine-tune
        fine_tune_label = QLabel("Fine-tune:")
        self.fine_tune_checkbox = QCheckBox()

        self.launch_button = QPushButton("Launch")
        self.launch_button.clicked.connect(self.launch_button_clicked)

        # Layout
        layout = QGridLayout()

        layout.addWidget(mode_label, 0, 0)
        layout.addWidget(self.mode_combo, 0, 1)

        layout.addWidget(model_name_label, 1, 0)
        layout.addWidget(self.model_name_combo, 1, 1)

        layout.addWidget(weights_label, 2, 0)
        layout.addWidget(self.weights_entry, 2, 1)
        layout.addWidget(self.weights_button, 2, 2)

        layout.addWidget(dataset_label, 3, 0)
        layout.addWidget(self.dataset_entry, 3, 1)
        layout.addWidget(self.dataset_button, 3, 2)

        layout.addWidget(output_label, 4, 0)
        layout.addWidget(self.output_entry, 4, 1)
        layout.addWidget(self.output_button, 4, 2)

        layout.addWidget(image_input_label, 5, 0)
        layout.addWidget(self.image_input_entry, 5, 1)

        layout.addWidget(epochs_label, 6, 0)
        layout.addWidget(self.epochs_entry, 6, 1)

        layout.addWidget(learning_rate_label, 7, 0)
        layout.addWidget(self.learning_rate_entry, 7, 1)

        layout.addWidget(save_images_label, 8, 0)
        layout.addWidget(self.save_images_checkbox, 8, 1)

        layout.addWidget(visualize_label, 9, 0)
        layout.addWidget(self.visualize_checkbox, 9, 1)

        layout.addWidget(pretrained_label, 10, 0)
        layout.addWidget(self.pretrained_checkbox, 10, 1)

        layout.addWidget(fine_tune_label, 11, 0)
        layout.addWidget(self.fine_tune_checkbox, 11, 1)

        layout.addWidget(self.launch_button, 12, 0, 1, 2)  # Span 1 row, 2 columns

        self.setLayout(layout)

    def open_weights_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.weights_entry.setText(directory)

    def open_dataset_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dataset_entry.setText(directory)

    def open_output_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.output_entry.setText(directory)

    def handle_mode_change(self, index):
        # Determine the selected mode
        selected_mode = self.mode_combo.itemText(index)

        # Disable/enable components based on the selected mode
        if selected_mode == "Train":
            self.model_name_combo.setEnabled(True)
            self.weights_entry.setEnabled(False)
            self.weights_button.setEnabled(False)
            self.dataset_entry.setEnabled(True)
            self.dataset_button.setEnabled(True)
            self.output_entry.setEnabled(True)
            self.output_button.setEnabled(True)
            self.save_images_checkbox.setEnabled(False)
            self.image_input_entry.setEnabled(False)
            self.visualize_checkbox.setEnabled(False)
            self.epochs_entry.setEnabled(True)
            self.pretrained_checkbox.setEnabled(True)
            self.learning_rate_entry.setEnabled(True)
            self.fine_tune_checkbox.setEnabled(True)

        elif selected_mode == "Local Inference":
            self.model_name_combo.setEnabled(True)
            self.weights_entry.setEnabled(True)
            self.weights_button.setEnabled(True)
            self.dataset_entry.setEnabled(True)
            self.dataset_button.setEnabled(True)
            self.output_entry.setEnabled(True)
            self.output_button.setEnabled(True)
            self.save_images_checkbox.setEnabled(True)
            self.image_input_entry.setEnabled(False)
            self.visualize_checkbox.setEnabled(True)
            self.epochs_entry.setEnabled(False)
            self.pretrained_checkbox.setEnabled(False)
            self.learning_rate_entry.setEnabled(False)
            self.fine_tune_checkbox.setEnabled(False)

        elif selected_mode == "Live Inference":
            self.model_name_combo.setEnabled(True)
            self.weights_entry.setEnabled(True)
            self.weights_button.setEnabled(True)
            self.dataset_entry.setEnabled(False)
            self.dataset_button.setEnabled(False)
            self.output_entry.setEnabled(False)
            self.output_button.setEnabled(False)
            self.save_images_checkbox.setEnabled(False)
            self.image_input_entry.setEnabled(True)
            self.visualize_checkbox.setEnabled(False)
            self.epochs_entry.setEnabled(False)
            self.pretrained_checkbox.setEnabled(False)
            self.learning_rate_entry.setEnabled(False)
            self.fine_tune_checkbox.setEnabled(False)

        elif selected_mode == "Metrics":
            self.model_name_combo.setEnabled(True)
            self.weights_entry.setEnabled(False)
            self.weights_button.setEnabled(False)
            self.dataset_entry.setEnabled(True)
            self.dataset_button.setEnabled(True)
            self.output_entry.setEnabled(True)
            self.output_button.setEnabled(True)
            self.save_images_checkbox.setEnabled(False)
            self.image_input_entry.setEnabled(False)
            self.visualize_checkbox.setEnabled(False)
            self.epochs_entry.setEnabled(True)
            self.pretrained_checkbox.setEnabled(True)
            self.learning_rate_entry.setEnabled(True)
            self.fine_tune_checkbox.setEnabled(True)

    def launch_button_clicked(self):
        self.model_name = self.model_name_combo.currentText()
        self.mode = self.mode_combo.currentText()
        self.weights_path = self.weights_entry.text()
        self.dataset_path = self.dataset_entry.text()
        self.output_path = self.output_entry.text()
        self.image_input_path = self.image_input_entry.text()
        self.epochs = self.epochs_entry.text()
        self.learning_rate = self.learning_rate_entry.text()
        self.save_images_checked = self.save_images_checkbox.isChecked()
        self.visualize_checked = self.visualize_checkbox.isChecked()
        self.pretrained_checked = self.pretrained_checkbox.isChecked()
        self.fine_tune_checked = self.fine_tune_checkbox.isChecked()

        message = f"Model: {self.model_name}\nMode: {self.mode}\nWeights Path: {self.weights_path}\nDataset Path: {self.dataset_path}\n" \
                  f"Output Path: {self.output_path}\nImage Input Path: {self.image_input_path}\nEpochs: {self.epochs}\n" \
                  f"Learning Rate: {self.learning_rate}\nSave Images: {self.save_images_checked}\nVisualize: {self.visualize_checked}\n" \
                  f"Pretrained: {self.pretrained_checked}\nFine-tune: {self.fine_tune_checked}"

        QMessageBox.information(self, "Launch Info", message)

        workspace = os.getcwd()

        print(f"El directorio de trabajo actual es: {workspace}")

        class_names = [name for name in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, name))]
        num_classes = len(class_names)

        model = ModelBuilder(
            name=self.model_name,
            pretrained=self.pretrained_checked,
            fine_tune=self.fine_tune_checked,
            num_classes=num_classes,
            model_name=self.model_name,
            weights=self.weights_path,
        )

        if self.mode.lower() == 'train':
            from ..models import Trainer
            from ..tools import DataPreparation

            data_prep = DataPreparation(
                root_dir=self.dataset_path,
                pretrained=self.pretrained_checked,
            )

            trainer = Trainer(
                data_preparation=data_prep,
                model=model,
                learning_rate=float(self.learning_rate),
                epochs=int(self.epochs),
            )

            trainer.run()

        elif self.mode.lower() == 'metrics':

            from ..inference import Metrics

            metrics = Metrics(
                model_builder=model,
                input_dir=self.input,
                output_dir=self.output
            )