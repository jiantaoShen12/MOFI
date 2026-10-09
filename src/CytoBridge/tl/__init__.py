def fit(*args, **kwargs):
    """Lazily dispatch to the public training entry point.

    Keeping this import lazy avoids the historical ``pl.plot`` ↔ ``tl.trainer``
    circular import while preserving ``CytoBridge.tl.fit(...)``.
    """
    from .fit import fit as _fit

    return _fit(*args, **kwargs)

from .analysis import *
from .analysis_dense_time import *
from .flow_matching import *
from .perturbation import *
