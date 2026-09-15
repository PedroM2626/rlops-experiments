"""SB3 import shim: SB3<2.10 imports cv2 via old gym path; on NumPy 2 the
bundled opencv build can fail with `_ARRAY_API not found`. Importing SB3
with a tiny cv2 stub (Atari wrappers unused here) keeps this study
runnable on the reporter's machine. Uses real cv2 when it imports."""
from __future__ import annotations

import importlib.util
import sys
import types


def ensure_sb3_importable() -> None:
    import sys as _sys
    try:
        import cv2 as _real  # noqa: F401
        _ = _real.__version__
        return
    except Exception:
        pass
    for name in ("cv2", "gym"):
        _sys.modules.pop(name, None)
    import types as _t
    import unittest.mock as _mock
    cv2 = _mock.MagicMock(name="cv2")
    cv2.__version__ = "0-stub"
    cv2.__spec__ = importlib.util.spec_from_loader("cv2", loader=None)
    _sys.modules["cv2"] = cv2
    try:
        import gymnasium as _gym
        gym_stub = _t.ModuleType("gym")
        gym_stub.__spec__ = importlib.util.spec_from_loader("gym",
                                                            loader=None)
        for attr in ("Env", "Wrapper", "ObservationWrapper",
                     "ActionWrapper", "RewardWrapper", "spaces"):
            if hasattr(_gym, attr):
                setattr(gym_stub, attr, getattr(_gym, attr))
        gym_stub.spaces = _gym.spaces
        _sys.modules["gym"] = gym_stub
    except Exception:
        pass

