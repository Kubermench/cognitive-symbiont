"""Comprehensive audit logging for security and compliance."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import hmac
import threading

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication & Authorization
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Data Access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # System Operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    CREDENTIAL_ROTATION = "credential_rotation"
    
    # LLM Operations
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    
    # Agent Operations
    AGENT_CREATE = "agent_create"
    AGENT_UPDATE = "agent_update"
    AGENT_DELETE = "agent_delete"
    AGENT_EXECUTE = "agent_execute"
    
    # Security Events
    SECURITY_ALERT = "security_alert"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PII_DETECTED = "pii_detected"
    PII_SANITIZED = "pii_sanitized"
    
    # Network Operations
    NETWORK_REQUEST = "network_request"
    NETWORK_RESPONSE = "network_response"
    WEBHOOK_SENT = "webhook_sent"
    
    # File Operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    
    # Database Operations
    DB_QUERY = "db_query"
    DB_UPDATE = "db_update"
    DB_DELETE = "db_delete"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    severity: AuditSeverity = AuditSeverity.LOW
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, db_path: Path, retention_days: int = 365):
        self.db_path = db_path
        self.retention_days = retention_days
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the audit database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    severity TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    resource TEXT,
                    action TEXT,
                    outcome TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s','now'))
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_session_id ON audit_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity);
                CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_events(resource);
                
                CREATE TABLE IF NOT EXISTS audit_retention (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_cleanup INTEGER NOT NULL,
                    events_deleted INTEGER DEFAULT 0
                );
            """)
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        timestamp = int(time.time() * 1000)  # milliseconds
        random_part = os.urandom(8).hex()
        return f"audit_{timestamp}_{random_part}"
    
    def log_event(self, 
                  event_type: AuditEventType,
                  message: str = "",
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  severity: AuditSeverity = AuditSeverity.LOW,
                  details: Optional[Dict[str, Any]] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  outcome: Optional[str] = None,
                  error_code: Optional[str] = None,
                  error_message: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log an audit event."""
        
        event_id = self._generate_event_id()
        timestamp = datetime.now()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            severity=severity,
            message=message,
            details=details or {},
            resource=resource,
            action=action,
            outcome=outcome,
            error_code=error_code,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        return self._store_event(event)
    
    def _store_event(self, event: AuditEvent) -> str:
        """Store an audit event in the database."""
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("""
                        INSERT INTO audit_events
                        (event_id, event_type, timestamp, user_id, session_id, source_ip,
                         user_agent, severity, message, details, resource, action, outcome,
                         error_code, error_message, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        int(event.timestamp.timestamp()),
                        event.user_id,
                        event.session_id,
                        event.source_ip,
                        event.user_agent,
                        event.severity.value,
                        event.message,
                        json.dumps(event.details),
                        event.resource,
                        event.action,
                        event.outcome,
                        event.error_code,
                        event.error_message,
                        json.dumps(event.metadata)
                    ))
                
                logger.debug("Logged audit event: %s", event.event_id)
                return event.event_id
                
            except Exception as e:
                logger.error("Failed to store audit event: %s", e)
                return ""
    
    def query_events(self,
                    event_types: Optional[List[AuditEventType]] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    severity: Optional[AuditSeverity] = None,
                    resource: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    limit: int = 1000,
                    offset: int = 0) -> List[AuditEvent]:
        """Query audit events with filters."""
        
        conditions = []
        params = []
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        
        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)
        
        if resource:
            conditions.append("resource = ?")
            params.append(resource)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(int(start_time.timestamp()))
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(int(end_time.timestamp()))
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT event_id, event_type, timestamp, user_id, session_id, source_ip,
                   user_agent, severity, message, details, resource, action, outcome,
                   error_code, error_message, metadata
            FROM audit_events
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        events = []
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    event_id, event_type, timestamp, user_id, session_id, source_ip, \
                    user_agent, severity, message, details, resource, action, outcome, \
                    error_code, error_message, metadata = row
                    
                    events.append(AuditEvent(
                        event_id=event_id,
                        event_type=AuditEventType(event_type),
                        timestamp=datetime.fromtimestamp(timestamp),
                        user_id=user_id,
                        session_id=session_id,
                        source_ip=source_ip,
                        user_agent=user_agent,
                        severity=AuditSeverity(severity),
                        message=message,
                        details=json.loads(details) if details else {},
                        resource=resource,
                        action=action,
                        outcome=outcome,
                        error_code=error_code,
                        error_message=error_message,
                        metadata=json.loads(metadata) if metadata else {}
                    ))
                    
        except Exception as e:
            logger.error("Failed to query audit events: %s", e)
        
        return events
    
    def get_event_statistics(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistics about audit events."""
        
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(int(start_time.timestamp()))
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(int(end_time.timestamp()))
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        stats = {}
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Total events
                cursor = conn.execute(f"SELECT COUNT(*) FROM audit_events {where_clause}", params)
                stats["total_events"] = cursor.fetchone()[0]
                
                # Events by type
                cursor = conn.execute(f"""
                    SELECT event_type, COUNT(*) 
                    FROM audit_events {where_clause}
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                """, params)
                stats["events_by_type"] = dict(cursor.fetchall())
                
                # Events by severity
                cursor = conn.execute(f"""
                    SELECT severity, COUNT(*) 
                    FROM audit_events {where_clause}
                    GROUP BY severity
                    ORDER BY COUNT(*) DESC
                """, params)
                stats["events_by_severity"] = dict(cursor.fetchall())
                
                # Events by user
                cursor = conn.execute(f"""
                    SELECT user_id, COUNT(*) 
                    FROM audit_events {where_clause}
                    WHERE user_id IS NOT NULL
                    GROUP BY user_id
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """, params)
                stats["events_by_user"] = dict(cursor.fetchall())
                
                # Recent events (last 24 hours)
                recent_cutoff = int((datetime.now().timestamp() - 86400))
                cursor = conn.execute(f"""
                    SELECT COUNT(*) FROM audit_events 
                    {where_clause} AND timestamp >= ?
                """, params + [recent_cutoff])
                stats["recent_events_24h"] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error("Failed to get audit statistics: %s", e)
            stats = {"error": str(e)}
        
        return stats
    
    def cleanup_old_events(self) -> int:
        """Remove events older than retention period."""
        cutoff_time = int((datetime.now().timestamp() - (self.retention_days * 86400)))
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp < ?", (cutoff_time,))
                count_before = cursor.fetchone()[0]
                
                conn.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff_time,))
                
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp < ?", (cutoff_time,))
                count_after = cursor.fetchone()[0]
                
                deleted_count = count_before - count_after
                
                # Record cleanup
                conn.execute("""
                    INSERT INTO audit_retention (last_cleanup, events_deleted)
                    VALUES (?, ?)
                """, (int(time.time()), deleted_count))
                
                logger.info("Cleaned up %d old audit events", deleted_count)
                return deleted_count
                
        except Exception as e:
            logger.error("Failed to cleanup old audit events: %s", e)
            return 0
    
    def export_events(self, 
                     output_path: Path,
                     event_types: Optional[List[AuditEventType]] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     format: str = "json") -> bool:
        """Export audit events to file."""
        
        events = self.query_events(
            event_types=event_types,
            start_time=start_time,
            end_time=end_time,
            limit=0  # No limit for export
        )
        
        try:
            if format.lower() == "json":
                export_data = []
                for event in events:
                    export_data.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "source_ip": event.source_ip,
                        "user_agent": event.user_agent,
                        "severity": event.severity.value,
                        "message": event.message,
                        "details": event.details,
                        "resource": event.resource,
                        "action": event.action,
                        "outcome": event.outcome,
                        "error_code": event.error_code,
                        "error_message": event.error_message,
                        "metadata": event.metadata
                    })
                
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "csv":
                import csv
                
                with output_path.open("w", newline="", encoding="utf-8") as f:
                    if events:
                        writer = csv.DictWriter(f, fieldnames=[
                            "event_id", "event_type", "timestamp", "user_id", "session_id",
                            "source_ip", "user_agent", "severity", "message", "resource",
                            "action", "outcome", "error_code", "error_message"
                        ])
                        writer.writeheader()
                        
                        for event in events:
                            writer.writerow({
                                "event_id": event.event_id,
                                "event_type": event.event_type.value,
                                "timestamp": event.timestamp.isoformat(),
                                "user_id": event.user_id or "",
                                "session_id": event.session_id or "",
                                "source_ip": event.source_ip or "",
                                "user_agent": event.user_agent or "",
                                "severity": event.severity.value,
                                "message": event.message,
                                "resource": event.resource or "",
                                "action": event.action or "",
                                "outcome": event.outcome or "",
                                "error_code": event.error_code or "",
                                "error_message": event.error_message or ""
                            })
            
            else:
                logger.error("Unsupported export format: %s", format)
                return False
            
            logger.info("Exported %d audit events to %s", len(events), output_path)
            return True
            
        except Exception as e:
            logger.error("Failed to export audit events: %s", e)
            return False


class AuditContext:
    """Context manager for audit logging."""
    
    def __init__(self, 
                 audit_logger: AuditLogger,
                 event_type: AuditEventType,
                 message: str = "",
                 **kwargs):
        self.audit_logger = audit_logger
        self.event_type = event_type
        self.message = message
        self.kwargs = kwargs
        self.event_id = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.event_id = self.audit_logger.log_event(
            event_type=self.event_type,
            message=self.message,
            **self.kwargs
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.event_id:
            duration = time.time() - self.start_time if self.start_time else 0
            
            if exc_type:
                # Log failure
                self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    message=f"Operation failed: {self.message}",
                    outcome="failure",
                    error_message=str(exc_val),
                    metadata={"duration_seconds": duration, "parent_event_id": self.event_id},
                    **self.kwargs
                )
            else:
                # Log success
                self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM_SUCCESS,
                    message=f"Operation completed: {self.message}",
                    outcome="success",
                    metadata={"duration_seconds": duration, "parent_event_id": self.event_id},
                    **self.kwargs
                )


def create_audit_logger(data_root: Path, retention_days: int = 365) -> AuditLogger:
    """Create an audit logger instance."""
    return AuditLogger(data_root / "audit.db", retention_days)


def audit_context(audit_logger: AuditLogger, 
                 event_type: AuditEventType, 
                 message: str = "", 
                 **kwargs) -> AuditContext:
    """Create an audit context for automatic success/failure logging."""
    return AuditContext(audit_logger, event_type, message, **kwargs)