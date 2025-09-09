from __future__ import annotations
import argparse, json, socket, threading
from typing import Dict, Any

from ..orchestrator import Orchestrator
from ..initiative import daemon as initiative
from ..ports import browser as browser_port
import yaml


def load_cfg(path: str = "./configs/config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def handle_request(msg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    try:
        if method == "ping":
            return {"id": mid, "result": {"pong": True}}
        if method == "run_cycle":
            goal = params.get("goal") or "Propose one 10-minute refactor for my repo"
            res = Orchestrator(cfg).cycle(goal=goal)
            return {"id": mid, "result": {"episode_id": res.get("episode_id"), "decision": res.get("decision")}}
        if method == "propose":
            res = initiative.propose_once(cfg, reason="mcp")
            return {"id": mid, "result": {"episode_id": res.get("episode_id"), "decision": res.get("decision")}}
        if method == "read_file":
            path = params.get("path")
            if not path:
                raise ValueError("path required")
            with open(path, "r", encoding="utf-8") as f:
                return {"id": mid, "result": {"content": f.read()}}
        if method == "browse_fetch":
            url = params.get("url")
            if not url:
                raise ValueError("url required")
            path = browser_port.fetch_to_artifact(url, cfg)
            return {"id": mid, "result": {"path": path}}
        return {"id": mid, "error": {"code": -32601, "message": "Method not found"}}
    except Exception as e:
        return {"id": mid, "error": {"code": -32000, "message": str(e)}}


def serve(host: str, port: int, cfg: Dict[str, Any]):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    print(f"[mcp] listening on {host}:{port} (newline-delimited JSON-RPC)")

    def client_thread(conn, addr):
        with conn:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception:
                        resp = {"id": None, "error": {"code": -32700, "message": "Parse error"}}
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                        continue
                    resp = handle_request(msg, cfg)
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=client_thread, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("[mcp] shutting down")


def main():
    ap = argparse.ArgumentParser(description="Minimal MCP-like JSON-RPC server (TCP, newline-delimited)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default="./configs/config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    serve(args.host, args.port, cfg)


if __name__ == "__main__":
    main()

