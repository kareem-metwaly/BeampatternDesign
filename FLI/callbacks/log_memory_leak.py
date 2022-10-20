"""implementation of logging callback to check memory leakage."""

import pickle as pkl
import shutil
import tracemalloc
import typing as t

from callbacks.callback import Callback
from models.base_model import Model
from trainer import Trainer
from utils.ProjectDirectory import ProjectDirectory


class LogMemoryLeak(Callback):
    """at start of each epoch checks the difference of memory allocation."""

    snapshot: tracemalloc.Snapshot

    def setup(self, trainer: Trainer, model: Model, stage: t.Optional[str] = None) -> None:
        """called in the beginning to define attributes that will be used
        later: objective is used to find the estimated beampattern."""
        super().__init__()
        shutil.rmtree(ProjectDirectory.logs_path("MemoryLeak"), ignore_errors=True)
        tracemalloc.start()

    def on_fit_start(self, trainer: Trainer, model: Model) -> None:
        """Called when fit begins."""
        self.snapshot = None

    def on_validation_epoch_end(self, trainer: Trainer, model: Model) -> None:
        """Called when the val epoch ends."""
        new_snapshot = tracemalloc.take_snapshot()
        with open(ProjectDirectory.logs_path("MemoryLeak").joinpath(f"dump_{trainer.current_epoch}.pkl"), "wb") as file:
            pkl.dump(new_snapshot, file)

        if self.snapshot is None:
            self.snapshot = new_snapshot
        else:
            top_stats = new_snapshot.compare_to(self.snapshot, "lineno")
            with open(
                ProjectDirectory.logs_path("MemoryLeak").joinpath(f"log_{trainer.current_epoch}.txt"), "w"
            ) as file:
                file.writelines(str(stat) for stat in top_stats)
            largest = new_snapshot.statistics("traceback")[0]
            with open(
                ProjectDirectory.logs_path("MemoryLeak").joinpath(f"traceback_{trainer.current_epoch}.txt"), "w"
            ) as file:
                file.writelines(largest.traceback.format())
