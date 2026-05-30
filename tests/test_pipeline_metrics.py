import numpy as np

from adsl.detection import HeuristicDetector


def test_detector_baseline_reduces_clean_risk():
    detector = HeuristicDetector(threshold=0.5)
    base = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    for _ in range(10):
        detector.update_baseline(base)
    risk = detector.score(base)
    assert risk < 0.05
