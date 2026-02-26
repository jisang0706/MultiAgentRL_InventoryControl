from pathlib import Path
import re
import shutil


class CheckpointBackupManager:
    def __init__(self, backup_dir: str = "", backup_every: int = 5):
        self.backup_dir = backup_dir.strip() if backup_dir else ""
        self.backup_every = max(int(backup_every), 1)
        self._last_backed_iteration = 0

    @property
    def enabled(self) -> bool:
        return bool(self.backup_dir)

    def resolve_checkpoint_path(self, save_result):
        if isinstance(save_result, str):
            return save_result

        checkpoint = getattr(save_result, "checkpoint", None)
        if checkpoint is not None:
            path = getattr(checkpoint, "path", None)
            if isinstance(path, str):
                return path

        if isinstance(save_result, dict):
            checkpoint = save_result.get("checkpoint")
            if isinstance(checkpoint, str):
                return checkpoint
            path = getattr(checkpoint, "path", None)
            if isinstance(path, str):
                return path

        text = str(save_result)
        match = re.search(r"path=([^\s,\)]+)", text)
        if match:
            return match.group(1).strip("'\"")

        match = re.search(r"(/tmp/tmp[\w\-/\.]+)", text)
        if match:
            return match.group(1).strip("'\"")

        return None

    def backup_checkpoint(self, checkpoint_path, iteration: int):
        if not self.enabled or not checkpoint_path:
            return

        src = Path(checkpoint_path)
        if not src.exists():
            print(f"[backup] checkpoint path not found: {src}")
            return

        backup_root = Path(self.backup_dir)
        backup_root.mkdir(parents=True, exist_ok=True)
        dst = backup_root / f"iter_{iteration:04d}_{src.name}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        self._last_backed_iteration = iteration
        print(f"[backup] copied checkpoint -> {dst}")

    def process_save_result(self, save_result, iteration: int):
        checkpoint_path = self.resolve_checkpoint_path(save_result)
        if self.enabled and checkpoint_path and (iteration % self.backup_every == 0):
            self.backup_checkpoint(checkpoint_path, iteration)
        return checkpoint_path

    def finalize(self, checkpoint_path, iteration: int):
        if not self.enabled or not checkpoint_path:
            return
        if self._last_backed_iteration == iteration:
            return
        self.backup_checkpoint(checkpoint_path, iteration)
