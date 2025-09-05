from __future__ import annotations
from typing import Dict, Any
class BaseAgent:
    name='base'
    def run(self, context: Dict[str,Any], memory):
        raise NotImplementedError
