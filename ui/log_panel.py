from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPlainTextEdit, QPushButton, QWidget


@dataclass
class LogEntry:
    time_text: str
    level: str
    source: str
    message: str


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[LogEntry] = []
        self.max_entries = 2000

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "INFO", "WARNING", "ERROR", "PROCESS", "DEBUG"])
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("SecondaryButton")
        layout.addWidget(self.level_filter)
        layout.addWidget(self.auto_scroll)
        layout.addStretch(1)
        layout.addWidget(self.clear_button)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(self.max_entries)
        self.text.setObjectName("LogText")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        from PyQt5.QtWidgets import QVBoxLayout

        vertical = QVBoxLayout()
        vertical.setContentsMargins(8, 8, 8, 8)
        vertical.addLayout(layout)
        vertical.addWidget(self.text, 1)
        root.addLayout(vertical)

        self.level_filter.currentTextChanged.connect(self.rebuild)
        self.clear_button.clicked.connect(self.clear)

    def append(self, level: str, message: str, source: str = "app") -> None:
        message = str(message).strip()
        if not message:
            return
        now = datetime.now().strftime("%H:%M:%S")
        entry = LogEntry(now, level.upper(), source, message)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        if self._accepts(entry):
            self.text.appendPlainText(self._format(entry))
            if self.auto_scroll.isChecked():
                self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def clear(self) -> None:
        self.entries.clear()
        self.text.clear()

    def rebuild(self) -> None:
        self.text.clear()
        for entry in self.entries:
            if self._accepts(entry):
                self.text.appendPlainText(self._format(entry))

    def _accepts(self, entry: LogEntry) -> bool:
        selected = self.level_filter.currentText()
        return selected == "ALL" or entry.level == selected

    @staticmethod
    def _format(entry: LogEntry) -> str:
        return f"{entry.time_text} | {entry.level:<7} | {entry.source:<8} | {entry.message}"
