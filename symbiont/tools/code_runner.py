from __future__ import annotations
import subprocess, tempfile, os, sys

def run_python_snippet(code: str, timeout: int = 3) -> str:
    prelude = "print('> sandbox start')\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td,'snippet.py')
        open(path,'w',encoding='utf-8').write(prelude+code)
        p = subprocess.run([sys.executable,'-S','-B',path],capture_output=True,text=True,timeout=timeout)
        return (p.stdout + p.stderr)[:2000]
