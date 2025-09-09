from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Capability(str, Enum):
    FS_WRITE = "fs_write"
    PROC_RUN = "proc_run"
    NET_READ = "net_read"


@dataclass
class Action:
    capability: Capability
    description: str
    preview: str  # e.g., script, diff, or command list


class Guard:
    def __init__(self, prompt: Callable[[Action], bool], auto_approve_safe: bool = False):
        self.prompt = prompt
        self.auto_approve_safe = auto_approve_safe

    def confirm(self, action: Action) -> bool:
        if self.auto_approve_safe and action.capability in {Capability.NET_READ}:
            return True
        return self.prompt(action)

