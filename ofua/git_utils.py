import git
from .workers import WorkerSignals


class GitProgress(git.remote.RemoteProgress):
    """Affiche la progression des opérations Git via les signaux PyQt."""

    def __init__(self, signals: WorkerSignals):
        super().__init__()
        self.signals = signals
        self.last_percentage = -1

    def update(self, op_code, cur_count, max_count=None, message=''):
        if max_count:
            percentage = int((cur_count / max_count) * 100)
            if percentage != self.last_percentage:
                self.signals.log.emit(
                    f"Git: {git.remote.RemoteProgress.OP_CODE_MAP.get(op_code, op_code)} - {cur_count}/{max_count} {message}"
                )
                self.last_percentage = percentage
        else:
            self.signals.log.emit(
                f"Git: {git.remote.RemoteProgress.OP_CODE_MAP.get(op_code, op_code)} - {cur_count} {message}"
            )
