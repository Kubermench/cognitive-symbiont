"""Credential rotation and management utilities.

This module provides secure credential rotation, storage, and management
capabilities for Symbiont.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..tools.secrets import SecretLoadError, load_secret


class CredentialManager:
    """Manages credential rotation and secure storage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.credentials_file = Path(config.get("credentials_file", "./data/credentials.json"))
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage."""
        key_file = Path(self.config.get("encryption_key_file", "./data/.encryption_key"))
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        
        # Generate new key
        key = Fernet.generate_key()
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Restrict permissions
        
        return key
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def store_credential(
        self,
        name: str,
        credential: str,
        metadata: Optional[Dict[str, Any]] = None,
        rotation_schedule: Optional[int] = None
    ) -> str:
        """Store a credential securely with optional rotation schedule."""
        credential_id = self._generate_credential_id()
        
        credential_data = {
            "id": credential_id,
            "name": name,
            "credential": self.fernet.encrypt(credential.encode()).decode(),
            "created_at": int(time.time()),
            "last_rotated": int(time.time()),
            "metadata": metadata or {},
            "rotation_schedule": rotation_schedule,  # seconds
            "version": 1
        }
        
        # Load existing credentials
        credentials = self._load_credentials()
        credentials[credential_id] = credential_data
        
        # Save credentials
        self._save_credentials(credentials)
        
        # Log credential creation
        self._log_credential_event("created", credential_id, name, metadata)
        
        return credential_id
    
    def retrieve_credential(self, credential_id: str) -> Optional[str]:
        """Retrieve a credential by ID."""
        credentials = self._load_credentials()
        credential_data = credentials.get(credential_id)
        
        if not credential_data:
            return None
        
        try:
            encrypted_credential = credential_data["credential"]
            decrypted = self.fernet.decrypt(encrypted_credential.encode())
            return decrypted.decode()
        except Exception:
            return None
    
    def rotate_credential(self, credential_id: str, new_credential: str) -> bool:
        """Rotate a credential to a new value."""
        credentials = self._load_credentials()
        credential_data = credentials.get(credential_id)
        
        if not credential_data:
            return False
        
        # Update credential
        credential_data["credential"] = self.fernet.encrypt(new_credential.encode()).decode()
        credential_data["last_rotated"] = int(time.time())
        credential_data["version"] += 1
        
        # Save updated credentials
        credentials[credential_id] = credential_data
        self._save_credentials(credentials)
        
        # Log rotation
        self._log_credential_event("rotated", credential_id, credential_data["name"], {
            "version": credential_data["version"]
        })
        
        return True
    
    def list_credentials(self) -> List[Dict[str, Any]]:
        """List all stored credentials (without sensitive data)."""
        credentials = self._load_credentials()
        
        result = []
        for cred_id, cred_data in credentials.items():
            result.append({
                "id": cred_id,
                "name": cred_data["name"],
                "created_at": cred_data["created_at"],
                "last_rotated": cred_data["last_rotated"],
                "rotation_schedule": cred_data.get("rotation_schedule"),
                "version": cred_data["version"],
                "metadata": cred_data.get("metadata", {})
            })
        
        return result
    
    def check_rotation_needed(self) -> List[Dict[str, Any]]:
        """Check which credentials need rotation."""
        credentials = self._load_credentials()
        now = int(time.time())
        needs_rotation = []
        
        for cred_id, cred_data in credentials.items():
            rotation_schedule = cred_data.get("rotation_schedule")
            if not rotation_schedule:
                continue
            
            last_rotated = cred_data.get("last_rotated", 0)
            if now - last_rotated >= rotation_schedule:
                needs_rotation.append({
                    "id": cred_id,
                    "name": cred_data["name"],
                    "last_rotated": last_rotated,
                    "rotation_schedule": rotation_schedule,
                    "overdue_by": now - last_rotated - rotation_schedule
                })
        
        return needs_rotation
    
    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        credentials = self._load_credentials()
        
        if credential_id not in credentials:
            return False
        
        credential_data = credentials[credential_id]
        del credentials[credential_id]
        
        self._save_credentials(credentials)
        
        # Log deletion
        self._log_credential_event("deleted", credential_id, credential_data["name"])
        
        return True
    
    def _generate_credential_id(self) -> str:
        """Generate a unique credential ID."""
        return secrets.token_urlsafe(16)
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from file."""
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_credentials(self, credentials: Dict[str, Any]) -> None:
        """Save credentials to file."""
        with open(self.credentials_file, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Restrict file permissions
        self.credentials_file.chmod(0o600)
    
    def _log_credential_event(
        self,
        event_type: str,
        credential_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log credential events for audit trail."""
        try:
            from ..initiative.state import get_state_store
            store = get_state_store(self.config)
            
            store.log_audit_event(
                event_type=f"credential_{event_type}",
                message=f"Credential '{name}' {event_type}",
                details={
                    "credential_id": credential_id,
                    "credential_name": name,
                    "metadata": metadata or {}
                }
            )
        except Exception:
            # Fallback to local logging if state store unavailable
            pass


class CredentialRotator:
    """Automated credential rotation service."""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
        self.running = False
    
    def start_rotation_service(self, check_interval: int = 3600) -> None:
        """Start the credential rotation service."""
        import threading
        
        self.running = True
        self.rotation_thread = threading.Thread(
            target=self._rotation_loop,
            args=(check_interval,),
            daemon=True
        )
        self.rotation_thread.start()
    
    def stop_rotation_service(self) -> None:
        """Stop the credential rotation service."""
        self.running = False
        if hasattr(self, 'rotation_thread'):
            self.rotation_thread.join(timeout=5)
    
    def _rotation_loop(self, check_interval: int) -> None:
        """Main rotation loop."""
        while self.running:
            try:
                self._check_and_rotate_credentials()
                time.sleep(check_interval)
            except Exception as e:
                print(f"Error in credential rotation: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _check_and_rotate_credentials(self) -> None:
        """Check and rotate credentials that need rotation."""
        needs_rotation = self.credential_manager.check_rotation_needed()
        
        for cred_info in needs_rotation:
            try:
                # Generate new credential (this would typically call an external service)
                new_credential = self._generate_new_credential(cred_info)
                
                if new_credential:
                    success = self.credential_manager.rotate_credential(
                        cred_info["id"],
                        new_credential
                    )
                    
                    if success:
                        print(f"Rotated credential: {cred_info['name']}")
                    else:
                        print(f"Failed to rotate credential: {cred_info['name']}")
                else:
                    print(f"Could not generate new credential for: {cred_info['name']}")
                    
            except Exception as e:
                print(f"Error rotating credential {cred_info['name']}: {e}")
    
    def _generate_new_credential(self, cred_info: Dict[str, Any]) -> Optional[str]:
        """Generate a new credential (implement based on your needs)."""
        # This is a placeholder - implement based on your credential generation needs
        # For example, generate API keys, passwords, etc.
        
        name = cred_info["name"]
        metadata = cred_info.get("metadata", {})
        
        if "api_key" in name.lower():
            return secrets.token_urlsafe(32)
        elif "password" in name.lower():
            return secrets.token_urlsafe(16)
        elif "token" in name.lower():
            return secrets.token_urlsafe(24)
        else:
            # Default: generate a random string
            return secrets.token_urlsafe(20)


class SecretValidator:
    """Validates secrets and credentials for security compliance."""
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        issues = []
        score = 0
        
        if len(password) < 8:
            issues.append("Password too short (minimum 8 characters)")
        else:
            score += 1
        
        if len(password) >= 12:
            score += 1
        
        if not any(c.isupper() for c in password):
            issues.append("Password should contain uppercase letters")
        else:
            score += 1
        
        if not any(c.islower() for c in password):
            issues.append("Password should contain lowercase letters")
        else:
            score += 1
        
        if not any(c.isdigit() for c in password):
            issues.append("Password should contain numbers")
        else:
            score += 1
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password):
            issues.append("Password should contain special characters")
        else:
            score += 1
        
        # Check for common patterns
        common_patterns = ["password", "123456", "qwerty", "admin", "user"]
        if any(pattern in password.lower() for pattern in common_patterns):
            issues.append("Password contains common patterns")
            score -= 1
        
        return {
            "score": score,
            "max_score": 6,
            "issues": issues,
            "is_strong": score >= 4 and not issues
        }
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> Dict[str, Any]:
        """Validate API key format."""
        issues = []
        
        if len(api_key) < 16:
            issues.append("API key too short (minimum 16 characters)")
        
        if len(api_key) > 128:
            issues.append("API key too long (maximum 128 characters)")
        
        # Check for valid characters (alphanumeric and common separators)
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=+")
        if not all(c in valid_chars for c in api_key):
            issues.append("API key contains invalid characters")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }
    
    @staticmethod
    def check_credential_exposure(credential: str) -> Dict[str, Any]:
        """Check if credential might be exposed (basic checks)."""
        issues = []
        
        # Check for common weak credentials
        weak_credentials = [
            "password", "123456", "admin", "test", "demo",
            "secret", "key", "token", "api", "user"
        ]
        
        if credential.lower() in weak_credentials:
            issues.append("Credential appears to be a common weak value")
        
        # Check for sequential patterns
        if credential.isdigit() and len(credential) > 3:
            if all(int(credential[i]) + 1 == int(credential[i+1]) for i in range(len(credential)-1)):
                issues.append("Credential appears to be sequential numbers")
        
        return {
            "is_exposed": len(issues) > 0,
            "issues": issues
        }