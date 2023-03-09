"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

from ..launchers import BaseLauncher
from ..models import Model

class SyncExecutor:
    """Synchronous executor for model inference.
    Args:
        model (Model): model for inference
        launcher (BaseLauncher): launcher to do inference
    """

    def __init__(self, model: Model, launcher: BaseLauncher) -> None:
        self.model = model
        self.launcher = launcher

    def run(self, frame):
        """Run demo using input stream (image, video stream, camera)."""

        preprocessed_data, meta = self.model.preprocess(frame)
        raw_output = self.launcher.infer_sync(preprocessed_data)
        output = self.model.postprocess(raw_output, meta)

        return output, meta
