import sys
from PyQt6.QtWidgets import QApplication
from ofua.ui_main import FalloutUpscalerApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FalloutUpscalerApp()
    window.show()
    sys.exit(app.exec())
