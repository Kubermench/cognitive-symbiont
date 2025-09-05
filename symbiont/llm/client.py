from __future__ import annotations
import subprocess, os
from typing import Dict

class LLMClient:
    def __init__(self, cfg: Dict):
        self.provider = (cfg or {}).get("provider","none")
        self.model = (cfg or {}).get("model","phi3:mini")
        self.cmd = (cfg or {}).get("cmd","")
        self.timeout = int((cfg or {}).get("timeout_seconds",25))

    def generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            try:
                out = subprocess.run(["ollama","generate","-m",self.model,"-p",prompt], capture_output=True, text=True, timeout=self.timeout)
                if out.returncode==0 and out.stdout.strip(): return out.stdout
            except Exception: pass
            try:
                p = subprocess.Popen(["ollama","run",self.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                s, e = p.communicate(prompt, timeout=self.timeout)
                if p.returncode==0 and s.strip(): return s
            except Exception: return ""
            return ""
        if self.provider == "cmd":
            if not self.cmd: return ""
            try:
                out = subprocess.run(self.cmd.replace("{prompt}", prompt.replace('"','\"')), shell=True, capture_output=True, text=True, timeout=self.timeout)
                return out.stdout
            except Exception: return ""
        return ""
