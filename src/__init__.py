"""
AdaptEval source package.

Core modules are imported explicitly by the pipeline when needed.
Keeping __init__.py lightweight prevents circular imports and avoids
loading the evaluation module unnecessarily.
"""

__all__ = []