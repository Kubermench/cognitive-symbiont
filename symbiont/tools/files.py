import os

def ensure_dirs(paths):
    for p in paths:
        if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)
