from pathlib import Path
from typing import List
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class ValidationDialog(QDialog):
    def __init__(self, originals: List[Path], upscaled: List[Path]):
        super().__init__()
        self.setWindowTitle("Valider l'upscaling")
        self.originals = originals
        self.upscaled = upscaled
        self.index = 0
        self.accepted = False
        self._init_ui()
        self._show_pair()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.image_label_orig = QLabel()
        self.image_label_up = QLabel()
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.image_label_orig)
        img_layout.addWidget(self.image_label_up)
        layout.addLayout(img_layout)
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Précédent")
        self.next_btn = QPushButton("Suivant")
        self.accept_btn = QPushButton("Accepter")
        self.reject_btn = QPushButton("Rejeter")
        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)
        self.accept_btn.clicked.connect(self.accept_action)
        self.reject_btn.clicked.connect(self.reject_action)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addWidget(self.accept_btn)
        btn_layout.addWidget(self.reject_btn)
        layout.addLayout(btn_layout)

    def _show_pair(self):
        orig = self.originals[self.index]
        up = self.upscaled[self.index]
        self.image_label_orig.setPixmap(QPixmap(str(orig)).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))
        self.image_label_up.setPixmap(QPixmap(str(up)).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self._show_pair()

    def next(self):
        if self.index < len(self.originals) - 1:
            self.index += 1
            self._show_pair()

    def accept_action(self):
        self.accepted = True
        self.accept()

    def reject_action(self):
        self.accepted = False
        self.reject()
