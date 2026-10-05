from .base import Detector
from .heuristic import HeuristicDetector


def load_detector(model_path: str | None = None) -> Detector:
    """Trained ONNX model if given, otherwise the training-free baseline."""
    if model_path:
        from .onnx_seg import OnnxSegmentationDetector

        return OnnxSegmentationDetector(model_path)
    return HeuristicDetector()


__all__ = ["Detector", "HeuristicDetector", "load_detector"]
