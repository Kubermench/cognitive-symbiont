"""Heartbeat management for distributed coordination.

This module provides automated heartbeat functionality for daemons and swarm workers,
ensuring that nodes can detect failures and coordinate properly in a distributed environment.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .state import get_state_store, resolve_node_id

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat management."""
    
    interval_seconds: float = 30.0
    stale_threshold_seconds: int = 300
    cleanup_interval_seconds: float = 120.0
    enabled: bool = True
    
    # Node capabilities and metadata
    capabilities: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    # Network discovery
    enable_discovery: bool = True
    discovery_port: int = 0


class HeartbeatManager:
    """Manages heartbeats for daemons and workers in a distributed environment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        heartbeat_cfg = self.config.get("heartbeat", {}) or {}
        
        self.heartbeat_config = HeartbeatConfig(
            interval_seconds=max(5.0, float(heartbeat_cfg.get("interval_seconds", 30.0) or 30.0)),
            stale_threshold_seconds=max(60, int(heartbeat_cfg.get("stale_threshold_seconds", 300) or 300)),
            cleanup_interval_seconds=max(30.0, float(heartbeat_cfg.get("cleanup_interval_seconds", 120.0) or 120.0)),
            enabled=bool(heartbeat_cfg.get("enabled", True)),
            capabilities=heartbeat_cfg.get("capabilities", {}) or {},
            version=heartbeat_cfg.get("version", "1.0"),
            enable_discovery=bool(heartbeat_cfg.get("enable_discovery", True)),
            discovery_port=int(heartbeat_cfg.get("discovery_port", 0) or 0),
        )
        
        self.store = get_state_store(self.config)
        self.node_id = resolve_node_id(self.config)
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        
        self._daemon_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._registered_workers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        logger.info(
            "HeartbeatManager initialized for node %s (interval: %.1fs, stale_threshold: %ds)",
            self.node_id,
            self.heartbeat_config.interval_seconds,
            self.heartbeat_config.stale_threshold_seconds,
        )
    
    def start(self) -> None:
        """Start the heartbeat manager."""
        if not self.heartbeat_config.enabled:
            logger.info("Heartbeat manager disabled by configuration")
            return
        
        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.warning("HeartbeatManager already running")
            return
        
        self._stop_event.clear()
        
        # Register this node for discovery
        if self.heartbeat_config.enable_discovery:
            try:
                self.store.register_node(
                    node_id=self.node_id,
                    hostname=self.hostname,
                    ip_address=self.ip_address,
                    port=self.heartbeat_config.discovery_port,
                    capabilities=self.heartbeat_config.capabilities,
                    version=self.heartbeat_config.version,
                )
                logger.info("Node %s registered for discovery", self.node_id)
            except Exception as exc:
                logger.warning("Failed to register node for discovery: %s", exc)
        
        # Start heartbeat thread
        self._daemon_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.node_id}",
            daemon=True,
        )
        self._daemon_thread.start()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name=f"heartbeat-cleanup-{self.node_id}",
            daemon=True,
        )
        self._cleanup_thread.start()
        
        logger.info("HeartbeatManager started for node %s", self.node_id)
    
    def stop(self) -> None:
        """Stop the heartbeat manager."""
        if not self.heartbeat_config.enabled:
            return
        
        self._stop_event.set()
        
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_thread.join(timeout=5.0)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("HeartbeatManager stopped for node %s", self.node_id)
    
    def register_worker(
        self,
        worker_id: str,
        goal: Optional[str] = None,
        priority: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a swarm worker for heartbeat management."""
        with self._lock:
            self._registered_workers[worker_id] = {
                "goal": goal,
                "priority": priority,
                "details": details or {},
                "registered_at": time.time(),
            }
        
        # Record the worker in the state store
        try:
            self.store.record_swarm_worker(
                worker_id=worker_id,
                node_id=self.node_id,
                status="active",
                goal=goal,
                variants=None,
                details=details,
            )
            logger.debug("Registered worker %s for heartbeats", worker_id)
        except Exception as exc:
            logger.warning("Failed to register worker %s: %s", worker_id, exc)
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a swarm worker from heartbeat management."""
        with self._lock:
            self._registered_workers.pop(worker_id, None)
        
        try:
            self.store.clear_swarm_worker(worker_id)
            logger.debug("Unregistered worker %s from heartbeats", worker_id)
        except Exception as exc:
            logger.warning("Failed to unregister worker %s: %s", worker_id, exc)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get the status of the entire cluster."""
        try:
            nodes = self.store.discover_nodes(self.heartbeat_config.stale_threshold_seconds)
            workers = self.store.list_swarm_workers()
            stale_daemons = self.store.get_stale_daemons(self.heartbeat_config.stale_threshold_seconds)
            stale_workers = self.store.get_stale_workers(self.heartbeat_config.stale_threshold_seconds)
            locks = self.store.list_coordination_locks()
            
            return {
                "cluster_size": len(nodes),
                "active_nodes": [n for n in nodes if n.get("status") == "active"],
                "total_workers": len(workers),
                "active_workers": [w for w in workers if w.get("status") == "active"],
                "stale_daemons": len(stale_daemons),
                "stale_workers": len(stale_workers),
                "coordination_locks": len(locks),
                "local_node_id": self.node_id,
                "local_workers": len(self._registered_workers),
            }
        except Exception as exc:
            logger.error("Failed to get cluster status: %s", exc)
            return {
                "error": str(exc),
                "local_node_id": self.node_id,
                "local_workers": len(self._registered_workers),
            }
    
    def acquire_leader_lock(self, purpose: str, ttl_seconds: int = 300) -> bool:
        """Acquire a leadership lock for coordination."""
        lock_name = f"leader_{purpose}"
        return self.store.acquire_coordination_lock(
            lock_name=lock_name,
            holder_node_id=self.node_id,
            purpose=f"Leadership for {purpose}",
            ttl_seconds=ttl_seconds,
            details={"acquired_by": self.hostname, "purpose": purpose},
        )
    
    def release_leader_lock(self, purpose: str) -> bool:
        """Release a leadership lock."""
        lock_name = f"leader_{purpose}"
        return self.store.release_coordination_lock(lock_name, self.node_id)
    
    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                # Send daemon heartbeat
                self.store.heartbeat_daemon(self.node_id)
                
                # Send worker heartbeats
                with self._lock:
                    for worker_id in list(self._registered_workers.keys()):
                        try:
                            self.store.heartbeat_worker(worker_id)
                        except Exception as exc:
                            logger.warning("Failed to send heartbeat for worker %s: %s", worker_id, exc)
                
                # Update node discovery
                if self.heartbeat_config.enable_discovery:
                    try:
                        self.store.register_node(
                            node_id=self.node_id,
                            hostname=self.hostname,
                            ip_address=self.ip_address,
                            port=self.heartbeat_config.discovery_port,
                            capabilities=self.heartbeat_config.capabilities,
                            version=self.heartbeat_config.version,
                        )
                    except Exception as exc:
                        logger.debug("Failed to update node discovery: %s", exc)
                
            except Exception as exc:
                logger.error("Error in heartbeat loop: %s", exc)
            
            # Wait for next heartbeat or stop signal
            if self._stop_event.wait(self.heartbeat_config.interval_seconds):
                break
    
    def _cleanup_loop(self) -> None:
        """Cleanup loop for stale entries."""
        while not self._stop_event.is_set():
            try:
                # Log stale daemons and workers for monitoring
                stale_daemons = self.store.get_stale_daemons(self.heartbeat_config.stale_threshold_seconds)
                stale_workers = self.store.get_stale_workers(self.heartbeat_config.stale_threshold_seconds)
                
                if stale_daemons:
                    logger.warning(
                        "Found %d stale daemons: %s",
                        len(stale_daemons),
                        [d["node_id"] for d in stale_daemons],
                    )
                
                if stale_workers:
                    logger.warning(
                        "Found %d stale workers: %s",
                        len(stale_workers),
                        [w["worker_id"] for w in stale_workers],
                    )
                
                # Clean up local worker registrations that are no longer valid
                with self._lock:
                    current_time = time.time()
                    to_remove = []
                    for worker_id, info in self._registered_workers.items():
                        # Remove workers that haven't been updated in a while
                        if current_time - info.get("registered_at", 0) > self.heartbeat_config.stale_threshold_seconds * 2:
                            to_remove.append(worker_id)
                    
                    for worker_id in to_remove:
                        logger.info("Cleaning up stale local worker registration: %s", worker_id)
                        self._registered_workers.pop(worker_id, None)
                        try:
                            self.store.clear_swarm_worker(worker_id)
                        except Exception:
                            pass
                
            except Exception as exc:
                logger.error("Error in cleanup loop: %s", exc)
            
            # Wait for next cleanup or stop signal
            if self._stop_event.wait(self.heartbeat_config.cleanup_interval_seconds):
                break
    
    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


# Global heartbeat manager instance
_heartbeat_manager: Optional[HeartbeatManager] = None
_manager_lock = threading.Lock()


def get_heartbeat_manager(config: Optional[Dict[str, Any]] = None) -> HeartbeatManager:
    """Get or create the global heartbeat manager."""
    global _heartbeat_manager
    
    with _manager_lock:
        if _heartbeat_manager is None:
            _heartbeat_manager = HeartbeatManager(config)
        return _heartbeat_manager


def start_heartbeat_manager(config: Optional[Dict[str, Any]] = None) -> HeartbeatManager:
    """Start the global heartbeat manager."""
    manager = get_heartbeat_manager(config)
    manager.start()
    return manager


def stop_heartbeat_manager() -> None:
    """Stop the global heartbeat manager."""
    global _heartbeat_manager
    
    with _manager_lock:
        if _heartbeat_manager:
            _heartbeat_manager.stop()
            _heartbeat_manager = None