"""Reproducibility helpers: seed every random source the framework uses."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow/Keras.

    ``deterministic_ops`` additionally forces TF to use deterministic kernels
    (exact bitwise reproducibility at a speed cost). Must be called before any
    TF op is executed to take full effect.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            tf.config.experimental.enable_op_determinism()
    except ImportError:  # realtime-only installs may not ship TF
        pass
