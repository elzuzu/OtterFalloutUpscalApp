import traceback
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    request_validation = pyqtSignal(str, str)


class TaskRunner(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.fn(self.signals, *self.args, **self.kwargs)
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(f"Erreur dans le worker: {e}\n{traceback.format_exc()}")

