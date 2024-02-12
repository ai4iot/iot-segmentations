from PyQt5.QtWidgets import QTextEdit


class LogWidget(QTextEdit):
    """A custom widget for logging messages."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def write(self, message):
        """Write a message to the log."""
        self.append(message)

    def flush(self):
        """Flush the log."""
        pass

