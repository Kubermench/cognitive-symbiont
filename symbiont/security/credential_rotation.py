"""Credential rotation and management utilities."""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


@dataclass
class CredentialInfo:
    """Information about a credential."""
    name: str
    type: str  # "api_key", "password", "token", "certificate"
    provider: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rotation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RotationPolicy:
    """Policy for credential rotation."""
    name: str
    credential_type: str
    rotation_interval_days: int
    warning_days: int = 7
    max_age_days: int = 365
    auto_rotate: bool = True
    require_manual_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CredentialStore:
    """Secure storage for credentials with rotation support."""
    
    def __init__(self, db_path: Path, master_key: Optional[bytes] = None):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate or load master key
        if master_key is None:
            master_key = self._load_or_generate_master_key()
        self.master_key = master_key
        self.cipher = Fernet(self.master_key)
        
        self._init_database()
    
    def _load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate new one."""
        key_path = self.db_path.parent / "master.key"
        
        if key_path.exists():
            try:
                return key_path.read_bytes()
            except Exception as e:
                logger.warning("Failed to load master key: %s", e)
        
        # Generate new master key
        key = Fernet.generate_key()
        try:
            key_path.write_bytes(key)
            key_path.chmod(0o600)  # Read/write for owner only
            logger.info("Generated new master key")
        except Exception as e:
            logger.warning("Failed to save master key: %s", e)
        
        return key
    
    def _init_database(self):
        """Initialize the credentials database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_value BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    last_used INTEGER,
                    rotation_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    active BOOLEAN DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS rotation_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    credential_type TEXT NOT NULL,
                    rotation_interval_days INTEGER NOT NULL,
                    warning_days INTEGER DEFAULT 7,
                    max_age_days INTEGER DEFAULT 365,
                    auto_rotate BOOLEAN DEFAULT 1,
                    require_manual_approval BOOLEAN DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at INTEGER NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS rotation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credential_name TEXT NOT NULL,
                    old_rotation_count INTEGER,
                    new_rotation_count INTEGER,
                    rotated_at INTEGER NOT NULL,
                    rotated_by TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_credentials_name ON credentials(name);
                CREATE INDEX IF NOT EXISTS idx_credentials_type ON credentials(type);
                CREATE INDEX IF NOT EXISTS idx_credentials_expires ON credentials(expires_at);
                CREATE INDEX IF NOT EXISTS idx_rotation_history_name ON rotation_history(credential_name);
            """)
    
    def store_credential(self, 
                        name: str, 
                        value: str, 
                        credential_type: str, 
                        provider: str,
                        expires_at: Optional[datetime] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a credential securely."""
        try:
            encrypted_value = self.cipher.encrypt(value.encode('utf-8'))
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO credentials 
                    (name, type, provider, encrypted_value, created_at, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    credential_type,
                    provider,
                    encrypted_value,
                    int(time.time()),
                    int(expires_at.timestamp()) if expires_at else None,
                    json.dumps(metadata or {})
                ))
            
            logger.info("Stored credential: %s", name)
            return True
            
        except Exception as e:
            logger.error("Failed to store credential %s: %s", name, e)
            return False
    
    def get_credential(self, name: str) -> Optional[str]:
        """Retrieve a credential by name."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT encrypted_value, last_used FROM credentials 
                    WHERE name = ? AND active = 1
                """, (name,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                encrypted_value, last_used = row
                
                # Update last used timestamp
                conn.execute("""
                    UPDATE credentials SET last_used = ? WHERE name = ?
                """, (int(time.time()), name))
                
                # Decrypt and return
                decrypted = self.cipher.decrypt(encrypted_value)
                return decrypted.decode('utf-8')
                
        except Exception as e:
            logger.error("Failed to retrieve credential %s: %s", name, e)
            return None
    
    def list_credentials(self) -> List[CredentialInfo]:
        """List all stored credentials."""
        credentials = []
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT name, type, provider, created_at, expires_at, 
                           last_used, rotation_count, metadata
                    FROM credentials WHERE active = 1
                    ORDER BY name
                """)
                
                for row in cursor.fetchall():
                    name, cred_type, provider, created_at, expires_at, last_used, rotation_count, metadata_json = row
                    
                    credentials.append(CredentialInfo(
                        name=name,
                        type=cred_type,
                        provider=provider,
                        created_at=datetime.fromtimestamp(created_at),
                        expires_at=datetime.fromtimestamp(expires_at) if expires_at else None,
                        last_used=datetime.fromtimestamp(last_used) if last_used else None,
                        rotation_count=rotation_count,
                        metadata=json.loads(metadata_json) if metadata_json else {}
                    ))
                    
        except Exception as e:
            logger.error("Failed to list credentials: %s", e)
        
        return credentials
    
    def rotate_credential(self, name: str, new_value: str, rotated_by: str = "system") -> bool:
        """Rotate a credential to a new value."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get current rotation count
                cursor = conn.execute("""
                    SELECT rotation_count FROM credentials WHERE name = ?
                """, (name,))
                row = cursor.fetchone()
                
                if not row:
                    logger.error("Credential %s not found", name)
                    return False
                
                old_rotation_count = row[0]
                new_rotation_count = old_rotation_count + 1
                
                # Update credential
                encrypted_value = self.cipher.encrypt(new_value.encode('utf-8'))
                conn.execute("""
                    UPDATE credentials 
                    SET encrypted_value = ?, rotation_count = ?, created_at = ?
                    WHERE name = ?
                """, (encrypted_value, new_rotation_count, int(time.time()), name))
                
                # Record rotation history
                conn.execute("""
                    INSERT INTO rotation_history 
                    (credential_name, old_rotation_count, new_rotation_count, 
                     rotated_at, rotated_by, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, old_rotation_count, new_rotation_count, 
                      int(time.time()), rotated_by, True))
                
                logger.info("Rotated credential: %s (rotation #%d)", name, new_rotation_count)
                return True
                
        except Exception as e:
            logger.error("Failed to rotate credential %s: %s", name, e)
            
            # Record failed rotation
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("""
                        INSERT INTO rotation_history 
                        (credential_name, old_rotation_count, new_rotation_count,
                         rotated_at, rotated_by, success, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (name, 0, 0, int(time.time()), rotated_by, False, str(e)))
            except Exception:
                pass
            
            return False
    
    def delete_credential(self, name: str) -> bool:
        """Delete a credential (soft delete)."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    UPDATE credentials SET active = 0 WHERE name = ?
                """, (name,))
            
            logger.info("Deleted credential: %s", name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete credential %s: %s", name, e)
            return False


class CredentialRotator:
    """Handles automatic credential rotation based on policies."""
    
    def __init__(self, credential_store: CredentialStore):
        self.store = credential_store
        self.policies: Dict[str, RotationPolicy] = {}
        self._load_policies()
    
    def _load_policies(self):
        """Load rotation policies from database."""
        try:
            with sqlite3.connect(str(self.store.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT name, credential_type, rotation_interval_days, 
                           warning_days, max_age_days, auto_rotate, 
                           require_manual_approval, metadata, created_at
                    FROM rotation_policies
                """)
                
                for row in cursor.fetchall():
                    name, cred_type, interval, warning, max_age, auto_rotate, manual_approval, metadata_json, created_at = row
                    
                    self.policies[name] = RotationPolicy(
                        name=name,
                        credential_type=cred_type,
                        rotation_interval_days=interval,
                        warning_days=warning,
                        max_age_days=max_age,
                        auto_rotate=bool(auto_rotate),
                        require_manual_approval=bool(manual_approval),
                        metadata=json.loads(metadata_json) if metadata_json else {}
                    )
                    
        except Exception as e:
            logger.error("Failed to load rotation policies: %s", e)
    
    def add_policy(self, policy: RotationPolicy) -> bool:
        """Add a new rotation policy."""
        try:
            with sqlite3.connect(str(self.store.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO rotation_policies
                    (name, credential_type, rotation_interval_days, warning_days,
                     max_age_days, auto_rotate, require_manual_approval, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    policy.name,
                    policy.credential_type,
                    policy.rotation_interval_days,
                    policy.warning_days,
                    policy.max_age_days,
                    policy.auto_rotate,
                    policy.require_manual_approval,
                    json.dumps(policy.metadata),
                    int(time.time())
                ))
            
            self.policies[policy.name] = policy
            logger.info("Added rotation policy: %s", policy.name)
            return True
            
        except Exception as e:
            logger.error("Failed to add rotation policy %s: %s", policy.name, e)
            return False
    
    def check_rotation_needed(self) -> List[Dict[str, Any]]:
        """Check which credentials need rotation."""
        rotation_needed = []
        now = datetime.now()
        
        for credential in self.store.list_credentials():
            # Find applicable policy
            policy = None
            for p in self.policies.values():
                if p.credential_type == credential.type:
                    policy = p
                    break
            
            if not policy:
                continue
            
            # Check if rotation is needed
            age_days = (now - credential.created_at).days
            
            if age_days >= policy.rotation_interval_days:
                rotation_needed.append({
                    "credential": credential,
                    "policy": policy,
                    "reason": "interval_expired",
                    "age_days": age_days
                })
            elif credential.expires_at and now >= credential.expires_at:
                rotation_needed.append({
                    "credential": credential,
                    "policy": policy,
                    "reason": "expired",
                    "age_days": age_days
                })
            elif age_days >= policy.max_age_days:
                rotation_needed.append({
                    "credential": credential,
                    "policy": policy,
                    "reason": "max_age_exceeded",
                    "age_days": age_days
                })
        
        return rotation_needed
    
    def rotate_credentials(self, dry_run: bool = False) -> Dict[str, Any]:
        """Rotate all credentials that need rotation."""
        results = {
            "rotated": [],
            "failed": [],
            "skipped": [],
            "dry_run": dry_run
        }
        
        rotation_needed = self.check_rotation_needed()
        
        for item in rotation_needed:
            credential = item["credential"]
            policy = item["policy"]
            
            if policy.require_manual_approval and not dry_run:
                results["skipped"].append({
                    "name": credential.name,
                    "reason": "requires_manual_approval"
                })
                continue
            
            if not policy.auto_rotate and not dry_run:
                results["skipped"].append({
                    "name": credential.name,
                    "reason": "auto_rotate_disabled"
                })
                continue
            
            if dry_run:
                results["rotated"].append({
                    "name": credential.name,
                    "reason": item["reason"],
                    "age_days": item["age_days"]
                })
                continue
            
            # Generate new credential value
            new_value = self._generate_credential_value(credential.type, credential.metadata)
            
            if new_value and self.store.rotate_credential(credential.name, new_value):
                results["rotated"].append({
                    "name": credential.name,
                    "reason": item["reason"],
                    "age_days": item["age_days"]
                })
            else:
                results["failed"].append({
                    "name": credential.name,
                    "reason": item["reason"],
                    "error": "Failed to generate or store new value"
                })
        
        return results
    
    def _generate_credential_value(self, credential_type: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Generate a new credential value based on type."""
        if credential_type == "api_key":
            # Generate random API key
            length = metadata.get("length", 32)
            return secrets.token_urlsafe(length)
        
        elif credential_type == "password":
            # Generate random password
            length = metadata.get("length", 16)
            return secrets.token_urlsafe(length)
        
        elif credential_type == "token":
            # Generate random token
            length = metadata.get("length", 24)
            return secrets.token_urlsafe(length)
        
        else:
            logger.warning("Unknown credential type: %s", credential_type)
            return None


def create_credential_store(data_root: Path, master_key: Optional[bytes] = None) -> CredentialStore:
    """Create a credential store instance."""
    return CredentialStore(data_root / "credentials.db", master_key)


def create_rotator(credential_store: CredentialStore) -> CredentialRotator:
    """Create a credential rotator instance."""
    return CredentialRotator(credential_store)