from __future__ import annotations
import os, re, time
def write_script(bullets, base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"apply_{int(time.time())}.sh")
    cmds=[]; 
    for b in bullets:
        m=re.search(r"\(cmd:\s*(.+?)\s*\)$", b)
        if m: cmds.append(m.group(1))
    body = ["# No explicit commands; fill in below","# echo 'do-something'"] if not cmds else cmds
    with open(path,"w",encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n"+ "\n".join(body)+ "\n")
    try: os.chmod(path,0o755)
    except Exception: pass
    return path
