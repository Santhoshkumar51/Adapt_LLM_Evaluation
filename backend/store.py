# backend/store.py
# Shared job store — lives here so no circular imports
from typing import Dict, Any

JOB_STORE: Dict[str, Any] = {}