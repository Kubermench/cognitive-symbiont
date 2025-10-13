"""HTTP server for Prometheus metrics endpoint.

This module provides a simple HTTP server that exposes Prometheus metrics
for monitoring and alerting.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from .metrics import get_metrics_response, update_system_metrics, update_llm_metrics, _load_budget_history, _snapshot_budget


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/health':
            self._serve_health()
        else:
            self._serve_404()
    
    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        try:
            metrics_data, content_type = get_metrics_response()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            self.wfile.write(metrics_data.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error generating metrics: {e}".encode('utf-8'))
    
    def _serve_health(self):
        """Serve health check endpoint."""
        try:
            # Simple health check
            health_data = '{"status": "healthy", "timestamp": ' + str(int(time.time())) + '}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(health_data.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(f'{{"status": "unhealthy", "error": "{e}"}}'.encode('utf-8'))
    
    def _serve_404(self):
        """Serve 404 for unknown paths."""
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass


class MetricsServer:
    """HTTP server for Prometheus metrics."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8001, config_path: str = "./configs/config.yaml"):
        self.host = host
        self.port = port
        self.config_path = config_path
        self.server = None
        self.thread = None
        self.running = False
    
    def start(self) -> None:
        """Start the metrics server."""
        if self.running:
            return
        
        try:
            self.server = HTTPServer((self.host, self.port), MetricsHandler)
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            self.running = True
            print(f"Metrics server started on {self.host}:{self.port}")
            print(f"Metrics endpoint: http://{self.host}:{self.port}/metrics")
            print(f"Health endpoint: http://{self.host}:{self.port}/health")
        except Exception as e:
            print(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the metrics server."""
        if not self.running:
            return
        
        self.running = False
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)
        print("Metrics server stopped")
    
    def _run_server(self) -> None:
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self.running:
                print(f"Metrics server error: {e}")
    
    def update_metrics(self) -> None:
        """Update metrics (called periodically)."""
        try:
            cfg = _load_budget_history(Path(self.config_path))
            data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent).resolve()
            
            # Update system metrics
            update_system_metrics(cfg)
            
            # Update LLM metrics
            history = _load_budget_history(data_root)
            update_llm_metrics(history)
            
        except Exception as e:
            print(f"Error updating metrics: {e}")


def main():
    """Main entry point for the metrics server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Symbiont Metrics Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--config", default="./configs/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    server = MetricsServer(args.host, args.port, args.config)
    
    try:
        server.start()
        
        # Update metrics periodically
        while True:
            time.sleep(5)
            server.update_metrics()
            
    except KeyboardInterrupt:
        print("\nShutting down metrics server...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()