"""PII (Personally Identifiable Information) sanitization utilities.

This module provides tools for detecting and sanitizing PII in transcripts,
artifacts, and other data to ensure privacy compliance.
"""

from __future__ import annotations

import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class PIIType(Enum):
    """Types of PII that can be detected and sanitized."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str


class PIIDetector:
    """Detects PII in text content."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            PIIType.PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                re.IGNORECASE
            ),
            PIIType.SSN: re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            PIIType.MAC_ADDRESS: re.compile(
                r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
            ),
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'
            ),
            PIIType.DRIVER_LICENSE: re.compile(
                r'\b[A-Z]\d{7,8}\b'
            ),
            PIIType.PASSPORT: re.compile(
                r'\b[A-Z]{1,2}\d{6,9}\b'
            ),
            PIIType.BANK_ACCOUNT: re.compile(
                r'\b\d{8,17}\b'
            ),
            PIIType.API_KEY: re.compile(
                r'\b[A-Za-z0-9]{20,}\b'
            ),
            PIIType.PASSWORD: re.compile(
                r'(?i)(?:password|pwd|pass)\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
                re.IGNORECASE
            ),
            PIIType.TOKEN: re.compile(
                r'(?i)(?:token|bearer|auth)\s*[:=]\s*["\']?([A-Za-z0-9._-]+)["\']?',
                re.IGNORECASE
            )
        }
        
        return patterns
    
    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Detect PII in text content."""
        matches = []
        
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                confidence = self._calculate_confidence(pii_type, value, text, match.start())
                
                if confidence > 0.5:  # Only include high-confidence matches
                    context = self._extract_context(text, match.start(), match.end())
                    
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        context=context
                    ))
        
        return matches
    
    def _calculate_confidence(self, pii_type: PIIType, value: str, text: str, position: int) -> float:
        """Calculate confidence score for a PII match."""
        base_confidence = 0.8
        
        # Adjust confidence based on context
        context = self._extract_context(text, position, position + len(value))
        
        # Check for common false positives
        if pii_type == PIIType.EMAIL:
            if any(word in context.lower() for word in ['example', 'test', 'demo', 'sample']):
                base_confidence *= 0.3
        
        elif pii_type == PIIType.PHONE:
            if any(word in context.lower() for word in ['extension', 'ext', 'fax']):
                base_confidence *= 0.5
        
        elif pii_type == PIIType.IP_ADDRESS:
            if any(word in context.lower() for word in ['localhost', '127.0.0.1', '192.168', '10.0', '172.16']):
                base_confidence *= 0.2
        
        return min(base_confidence, 1.0)
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Extract context around a PII match."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]


class PIISanitizer:
    """Sanitizes PII in text content."""
    
    def __init__(self, replacement_strategy: str = "hash"):
        self.replacement_strategy = replacement_strategy
        self.detector = PIIDetector()
        self.salt = "symbiont_pii_salt_2024"  # In production, use a secure random salt
    
    def sanitize_text(self, text: str, preserve_format: bool = True) -> Tuple[str, List[PIIMatch]]:
        """Sanitize PII in text content."""
        matches = self.detector.detect_pii(text)
        sanitized_text = text
        
        # Sort matches by position (reverse order to maintain positions)
        matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        for match in matches:
            replacement = self._generate_replacement(match, preserve_format)
            sanitized_text = (
                sanitized_text[:match.start_pos] + 
                replacement + 
                sanitized_text[match.end_pos:]
            )
        
        return sanitized_text, matches
    
    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 10) -> Tuple[Dict[str, Any], List[PIIMatch]]:
        """Sanitize PII in dictionary data."""
        all_matches = []
        sanitized_data = self._sanitize_dict_recursive(data, all_matches, max_depth)
        return sanitized_data, all_matches
    
    def _sanitize_dict_recursive(
        self, 
        data: Any, 
        all_matches: List[PIIMatch], 
        max_depth: int
    ) -> Any:
        """Recursively sanitize dictionary data."""
        if max_depth <= 0:
            return data
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                sanitized[key] = self._sanitize_dict_recursive(value, all_matches, max_depth - 1)
            return sanitized
        
        elif isinstance(data, list):
            return [
                self._sanitize_dict_recursive(item, all_matches, max_depth - 1)
                for item in data
            ]
        
        elif isinstance(data, str):
            sanitized_text, matches = self.sanitize_text(data)
            all_matches.extend(matches)
            return sanitized_text
        
        else:
            return data
    
    def _generate_replacement(self, match: PIIMatch, preserve_format: bool = True) -> str:
        """Generate replacement text for a PII match."""
        if self.replacement_strategy == "hash":
            return self._hash_replacement(match, preserve_format)
        elif self.replacement_strategy == "mask":
            return self._mask_replacement(match, preserve_format)
        elif self.replacement_strategy == "remove":
            return ""
        else:
            return "[REDACTED]"
    
    def _hash_replacement(self, match: PIIMatch, preserve_format: bool = True) -> str:
        """Generate hash-based replacement."""
        # Create a deterministic hash
        hash_input = f"{match.value}{self.salt}{match.pii_type.value}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        if preserve_format:
            # Try to preserve the format
            if match.pii_type == PIIType.EMAIL:
                return f"user{hash_value}@example.com"
            elif match.pii_type == PIIType.PHONE:
                return f"***-***-{hash_value[:4]}"
            elif match.pii_type == PIIType.SSN:
                return f"***-**-{hash_value[:4]}"
            elif match.pii_type == PIIType.CREDIT_CARD:
                return f"****-****-****-{hash_value[:4]}"
            else:
                return f"[{match.pii_type.value.upper()}:{hash_value}]"
        else:
            return f"[{match.pii_type.value.upper()}:{hash_value}]"
    
    def _mask_replacement(self, match: PIIMatch, preserve_format: bool = True) -> str:
        """Generate mask-based replacement."""
        if match.pii_type == PIIType.EMAIL:
            local, domain = match.value.split('@', 1)
            return f"{'*' * len(local)}@{domain}"
        elif match.pii_type == PIIType.PHONE:
            return f"***-***-{match.value[-4:]}"
        elif match.pii_type == PIIType.SSN:
            return f"***-**-{match.value[-4:]}"
        elif match.pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{match.value[-4:]}"
        else:
            return "*" * len(match.value)


class PIIAuditor:
    """Audits PII detection and sanitization for compliance."""
    
    def __init__(self):
        self.detector = PIIDetector()
        self.sanitizer = PIISanitizer()
    
    def audit_content(self, content: str) -> Dict[str, Any]:
        """Audit content for PII compliance."""
        matches = self.detector.detect_pii(content)
        
        # Group matches by type
        matches_by_type = {}
        for match in matches:
            pii_type = match.pii_type.value
            if pii_type not in matches_by_type:
                matches_by_type[pii_type] = []
            matches_by_type[pii_type].append({
                "value": match.value,
                "confidence": match.confidence,
                "context": match.context
            })
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(matches)
        
        return {
            "total_matches": len(matches),
            "matches_by_type": matches_by_type,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "recommendations": self._get_recommendations(matches)
        }
    
    def _calculate_risk_score(self, matches: List[PIIMatch]) -> float:
        """Calculate overall risk score for PII matches."""
        if not matches:
            return 0.0
        
        # Weight different PII types by sensitivity
        sensitivity_weights = {
            PIIType.SSN: 1.0,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.DRIVER_LICENSE: 0.8,
            PIIType.PASSPORT: 0.8,
            PIIType.BANK_ACCOUNT: 0.7,
            PIIType.PASSWORD: 0.9,
            PIIType.API_KEY: 0.8,
            PIIType.TOKEN: 0.7,
            PIIType.EMAIL: 0.3,
            PIIType.PHONE: 0.4,
            PIIType.NAME: 0.5,
            PIIType.ADDRESS: 0.6,
            PIIType.DATE_OF_BIRTH: 0.6,
            PIIType.IP_ADDRESS: 0.2,
            PIIType.MAC_ADDRESS: 0.1
        }
        
        total_score = 0.0
        for match in matches:
            weight = sensitivity_weights.get(match.pii_type, 0.5)
            total_score += weight * match.confidence
        
        return min(total_score / len(matches), 1.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level based on score."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_recommendations(self, matches: List[PIIMatch]) -> List[str]:
        """Get recommendations based on detected PII."""
        recommendations = []
        
        if not matches:
            return ["No PII detected - content appears clean"]
        
        # Count PII types
        pii_counts = {}
        for match in matches:
            pii_type = match.pii_type.value
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
        
        # Generate recommendations
        if PIIType.SSN.value in pii_counts:
            recommendations.append("CRITICAL: SSN detected - immediate sanitization required")
        
        if PIIType.CREDIT_CARD.value in pii_counts:
            recommendations.append("HIGH: Credit card numbers detected - sanitize before storage")
        
        if PIIType.PASSWORD.value in pii_counts:
            recommendations.append("HIGH: Passwords detected - remove or hash immediately")
        
        if PIIType.API_KEY.value in pii_counts:
            recommendations.append("MEDIUM: API keys detected - consider rotation")
        
        if len(matches) > 10:
            recommendations.append("HIGH: Large number of PII matches - comprehensive review needed")
        
        return recommendations


def sanitize_transcript(transcript_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PIIMatch]]:
    """Convenience function to sanitize transcript data."""
    sanitizer = PIISanitizer()
    return sanitizer.sanitize_dict(transcript_data)


def audit_transcript(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to audit transcript data for PII."""
    auditor = PIIAuditor()
    
    # Convert to string for analysis
    content = json.dumps(transcript_data, indent=2)
    return auditor.audit_content(content)