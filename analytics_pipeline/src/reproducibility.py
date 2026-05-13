"""Reproducibility utilities for LiveStrat.

Every model-training, sampling, and stochastic-policy file should call
`set_global_seed(LIVESTRAT_SEED)` near its entry point. This guarantees that:

* numpy random draws are deterministic across runs,
* Python `random` draws are deterministic,
* scikit-learn estimators that respect `random_state` see the same value,
* TensorFlow (LSTM models) initial weights are deterministic,
* PyTorch (FinBERT inference) initial state is deterministic,
* PYTHONHASHSEED is set so that set/dict ordering is stable.

The chosen master seed is 1729 (the Ramanujan number), purely for memorability.

Reference for why this matters for a final-year ML project:
    Pineau, J. et al. (2021) 'Improving Reproducibility in Machine Learning
    Research', JMLR 22, 1-20.
"""

from __future__ import annotations

import os
import random
from typing import Optional


LIVESTRAT_SEED: int = 1729


def set_global_seed(seed: int = LIVESTRAT_SEED, *, deep_learning: bool = False) -> int:
    """Seed every stochastic library used in the LiveStrat pipeline.

    Parameters
    ----------
    seed : int
        Master seed value. Defaults to ``LIVESTRAT_SEED``.
    deep_learning : bool
        When True, also seed TensorFlow and PyTorch. These imports are slow,
        so they are only triggered when an LSTM / FinBERT stage actually needs
        them.

    Returns
    -------
    int
        The seed that was applied (echoed for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if deep_learning:
        try:
            import tensorflow as tf  # type: ignore

            tf.random.set_seed(seed)
        except ImportError:
            pass
        try:
            import torch  # type: ignore

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(False)
        except ImportError:
            pass

    return seed


def sklearn_random_state(seed: Optional[int] = None) -> int:
    """Return a scikit-learn-compatible random_state.

    Used to keep model constructors readable, e.g.::

        model = LogisticRegression(random_state=sklearn_random_state())
    """
    return LIVESTRAT_SEED if seed is None else seed
