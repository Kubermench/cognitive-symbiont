from __future__ import annotations
from typing import List
import math, numpy as np

def cheap_embed(texts: List[str]) -> List[List[float]]:
    vecs = []
    for t in texts:
        h=[0]*64
        for i,ch in enumerate(t.encode('utf-8')):
            h[i%64]=(h[i%64]+ch)%997
        norm = math.sqrt(sum(v*v for v in h)) or 1.0
        vecs.append([v/norm for v in h])
    return vecs

def cosine(a,b):
    a=np.array(a); b=np.array(b)
    denom = (float(np.linalg.norm(a))*float(np.linalg.norm(b))) or 1.0
    return float(np.dot(a,b)/denom)
