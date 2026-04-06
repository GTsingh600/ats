"""Import utilities for relative/absolute import compatibility."""

from __future__ import annotations

from typing import Any


def safe_import(relative_import_func, absolute_import_func) -> Any:
    """
    Attempt relative import, fall back to absolute import.
    
    This eliminates boilerplate try/except blocks across modules.
    
    Args:
        relative_import_func: Callable that performs relative import (e.g., from .module import ...)
        absolute_import_func: Callable that performs absolute import (e.g., from module import ...)
    
    Returns:
        The imported module or object
        
    Example:
        models = safe_import(
            lambda: __import__('.models', fromlist=['FlightRecord']).FlightRecord,
            lambda: __import__('models', fromlist=['FlightRecord']).FlightRecord,
        )
    """
    try:
        return relative_import_func()
    except ImportError:
        return absolute_import_func()


# Convenience exports for models module
def get_models():
    """Get models module with automatic relative/absolute import fallback."""
    try:
        from . import models as _models
        return _models
    except ImportError:
        import models as _models
        return _models


def get_constants():
    """Get constants module with automatic relative/absolute import fallback."""
    try:
        from . import constants as _constants
        return _constants
    except ImportError:
        import constants as _constants
        return _constants
