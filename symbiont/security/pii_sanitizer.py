"""PII detection and sanitization utilities."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    UUID = "uuid"


@dataclass
class PIIDetection:
    """A detected PII instance."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""


@dataclass
class SanitizationResult:
    """Result of PII sanitization."""
    original_text: str
    sanitized_text: str
    detections: List[PIIDetection]
    sanitization_map: Dict[str, str]  # original -> sanitized mapping


class PIIDetector:
    """Detects personally identifiable information in text."""
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.context_window = 50  # Characters before/after match for context
    
    def _build_patterns(self) -> Dict[PIIType, List[re.Pattern]]:
        """Build regex patterns for PII detection."""
        patterns = {
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
            ],
            
            PIIType.PHONE: [
                re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
                re.compile(r'\b(?:\+?1[-.\s]?)?([0-9]{3})[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
                re.compile(r'\b(?:\+?[1-9]\d{1,14})\b')  # International format
            ],
            
            PIIType.SSN: [
                re.compile(r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'),
                re.compile(r'\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b')
            ],
            
            PIIType.CREDIT_CARD: [
                re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
                re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b')
            ],
            
            PIIType.IP_ADDRESS: [
                re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
                re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')  # IPv6
            ],
            
            PIIType.MAC_ADDRESS: [
                re.compile(r'\b(?:[0-9a-fA-F]{2}[:-]){5}(?:[0-9a-fA-F]{2})\b')
            ],
            
            PIIType.NAME: [
                re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', re.IGNORECASE),
                re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')  # First Last
            ],
            
            PIIType.ADDRESS: [
                re.compile(r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl|Court|Ct)\b', re.IGNORECASE)
            ],
            
            PIIType.DATE_OF_BIRTH: [
                re.compile(r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'),
                re.compile(r'\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2}\b')
            ],
            
            PIIType.PASSPORT: [
                re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
                re.compile(r'\b\d{9}\b')
            ],
            
            PIIType.DRIVER_LICENSE: [
                re.compile(r'\b[A-Z]\d{7,8}\b'),
                re.compile(r'\b\d{8,9}\b')
            ],
            
            PIIType.BANK_ACCOUNT: [
                re.compile(r'\b\d{8,17}\b')  # Generic bank account pattern
            ],
            
            PIIType.API_KEY: [
                re.compile(r'\b[A-Za-z0-9]{20,}\b'),  # Generic API key pattern
                re.compile(r'\b(?:sk|pk)_[A-Za-z0-9]{20,}\b'),  # Stripe-style
                re.compile(r'\b[A-Za-z0-9]{32,}\b')  # Common API key length
            ],
            
            PIIType.PASSWORD: [
                re.compile(r'\b(?:password|pwd|pass)\s*[:=]\s*["\']?[^"\'\s]+["\']?', re.IGNORECASE),
                re.compile(r'\b(?:secret|key|token)\s*[:=]\s*["\']?[^"\'\s]+["\']?', re.IGNORECASE)
            ],
            
            PIIType.TOKEN: [
                re.compile(r'\b[A-Za-z0-9+/]{20,}={0,2}\b'),  # Base64-like tokens
                re.compile(r'\b[A-Za-z0-9]{20,}\b')  # Generic token pattern
            ],
            
            PIIType.UUID: [
                re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')
            ]
        }
        
        return patterns
    
    def detect_pii(self, text: str) -> List[PIIDetection]:
        """Detect PII in text."""
        detections = []
        
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(pii_type, value, text, start_pos, end_pos)
                    
                    # Get context around the match
                    context_start = max(0, start_pos - self.context_window)
                    context_end = min(len(text), end_pos + self.context_window)
                    context = text[context_start:context_end]
                    
                    detections.append(PIIDetection(
                        pii_type=pii_type,
                        value=value,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        context=context
                    ))
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlapping_detections(detections)
        
        return detections
    
    def _calculate_confidence(self, pii_type: PIIType, value: str, text: str, start: int, end: int) -> float:
        """Calculate confidence score for a PII detection."""
        base_confidence = 0.5
        
        # Adjust based on PII type specificity
        if pii_type in [PIIType.SSN, PIIType.CREDIT_CARD, PIIType.UUID]:
            base_confidence = 0.9
        elif pii_type in [PIIType.EMAIL, PIIType.IP_ADDRESS, PIIType.MAC_ADDRESS]:
            base_confidence = 0.8
        elif pii_type in [PIIType.PHONE, PIIType.DATE_OF_BIRTH]:
            base_confidence = 0.7
        elif pii_type in [PIIType.API_KEY, PIIType.TOKEN]:
            base_confidence = 0.6
        elif pii_type in [PIIType.NAME, PIIType.ADDRESS]:
            base_confidence = 0.4
        
        # Adjust based on context clues
        context_start = max(0, start - 20)
        context_end = min(len(text), end + 20)
        context = text[context_start:context_end].lower()
        
        context_clues = {
            PIIType.EMAIL: ['email', 'mail', '@', 'contact'],
            PIIType.PHONE: ['phone', 'call', 'mobile', 'telephone'],
            PIIType.SSN: ['ssn', 'social security', 'tax id'],
            PIIType.CREDIT_CARD: ['card', 'credit', 'visa', 'mastercard'],
            PIIType.ADDRESS: ['address', 'street', 'avenue', 'road'],
            PIIType.NAME: ['name', 'first', 'last', 'mr', 'mrs', 'ms'],
            PIIType.DATE_OF_BIRTH: ['birth', 'born', 'dob', 'age']
        }
        
        if pii_type in context_clues:
            for clue in context_clues[pii_type]:
                if clue in context:
                    base_confidence += 0.1
                    break
        
        return min(1.0, base_confidence)
    
    def _remove_overlapping_detections(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping detections, keeping the highest confidence ones."""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            # Check if this detection overlaps with any already accepted detection
            overlaps = False
            for accepted in filtered:
                if (detection.start_pos < accepted.end_pos and 
                    detection.end_pos > accepted.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(detection)
        
        return filtered


class PIISanitizer:
    """Sanitizes PII in text by replacing with safe alternatives."""
    
    def __init__(self, detector: Optional[PIIDetector] = None):
        self.detector = detector or PIIDetector()
        self.sanitization_map: Dict[str, str] = {}
    
    def sanitize_text(self, text: str, 
                     replacement_strategy: str = "hash",
                     preserve_format: bool = True,
                     min_confidence: float = 0.5) -> SanitizationResult:
        """Sanitize PII in text."""
        
        detections = self.detector.detect_pii(text)
        
        # Filter by confidence
        detections = [d for d in detections if d.confidence >= min_confidence]
        
        # Sort by position (reverse order to avoid index shifting)
        detections.sort(key=lambda d: d.start_pos, reverse=True)
        
        sanitized_text = text
        sanitization_map = {}
        
        for detection in detections:
            original_value = detection.value
            sanitized_value = self._generate_replacement(
                original_value, 
                detection.pii_type, 
                replacement_strategy,
                preserve_format
            )
            
            # Replace in text
            sanitized_text = (sanitized_text[:detection.start_pos] + 
                            sanitized_value + 
                            sanitized_text[detection.end_pos:])
            
            # Track mapping
            sanitization_map[original_value] = sanitized_value
            self.sanitization_map[original_value] = sanitized_value
        
        return SanitizationResult(
            original_text=text,
            sanitized_text=sanitized_text,
            detections=detections,
            sanitization_map=sanitization_map
        )
    
    def _generate_replacement(self, 
                            original: str, 
                            pii_type: PIIType, 
                            strategy: str,
                            preserve_format: bool) -> str:
        """Generate a replacement for PII."""
        
        if strategy == "hash":
            # Use hash of original value for consistency
            hash_obj = hashlib.sha256(original.encode())
            hash_hex = hash_obj.hexdigest()[:8]
            return f"[{pii_type.value.upper()}_{hash_hex}]"
        
        elif strategy == "mask":
            # Mask with asterisks, preserving format
            if preserve_format:
                if pii_type == PIIType.EMAIL:
                    local, domain = original.split('@', 1)
                    return f"{local[0]}***@{domain}"
                elif pii_type == PIIType.PHONE:
                    return re.sub(r'\d', '*', original)
                elif pii_type == PIIType.SSN:
                    return f"***-**-{original[-4:]}"
                elif pii_type == PIIType.CREDIT_CARD:
                    return f"****-****-****-{original[-4:]}"
                else:
                    return "*" * len(original)
            else:
                return "*" * len(original)
        
        elif strategy == "remove":
            return "[REDACTED]"
        
        elif strategy == "replace":
            # Replace with generic placeholder
            replacements = {
                PIIType.EMAIL: "[EMAIL]",
                PIIType.PHONE: "[PHONE]",
                PIIType.SSN: "[SSN]",
                PIIType.CREDIT_CARD: "[CREDIT_CARD]",
                PIIType.IP_ADDRESS: "[IP_ADDRESS]",
                PIIType.MAC_ADDRESS: "[MAC_ADDRESS]",
                PIIType.NAME: "[NAME]",
                PIIType.ADDRESS: "[ADDRESS]",
                PIIType.DATE_OF_BIRTH: "[DOB]",
                PIIType.PASSPORT: "[PASSPORT]",
                PIIType.DRIVER_LICENSE: "[DL]",
                PIIType.BANK_ACCOUNT: "[ACCOUNT]",
                PIIType.API_KEY: "[API_KEY]",
                PIIType.PASSWORD: "[PASSWORD]",
                PIIType.TOKEN: "[TOKEN]",
                PIIType.UUID: "[UUID]"
            }
            return replacements.get(pii_type, "[PII]")
        
        else:
            return "[REDACTED]"
    
    def sanitize_json(self, data: Any, 
                     replacement_strategy: str = "hash",
                     min_confidence: float = 0.5) -> Tuple[Any, Dict[str, str]]:
        """Sanitize PII in JSON data."""
        
        if isinstance(data, str):
            result = self.sanitize_text(data, replacement_strategy, min_confidence=min_confidence)
            return result.sanitized_text, result.sanitization_map
        
        elif isinstance(data, dict):
            sanitized = {}
            all_mappings = {}
            
            for key, value in data.items():
                # Check if key suggests PII
                key_lower = key.lower()
                if any(pii in key_lower for pii in ['email', 'phone', 'ssn', 'name', 'address', 'password', 'token', 'key']):
                    if isinstance(value, str):
                        result = self.sanitize_text(value, replacement_strategy, min_confidence=min_confidence)
                        sanitized[key] = result.sanitized_text
                        all_mappings.update(result.sanitization_map)
                    else:
                        sanitized[key] = value
                else:
                    sanitized_value, mappings = self.sanitize_json(value, replacement_strategy, min_confidence)
                    sanitized[key] = sanitized_value
                    all_mappings.update(mappings)
            
            return sanitized, all_mappings
        
        elif isinstance(data, list):
            sanitized = []
            all_mappings = {}
            
            for item in data:
                sanitized_item, mappings = self.sanitize_json(item, replacement_strategy, min_confidence)
                sanitized.append(sanitized_item)
                all_mappings.update(mappings)
            
            return sanitized, all_mappings
        
        else:
            return data, {}


def create_pii_detector() -> PIIDetector:
    """Create a PII detector instance."""
    return PIIDetector()


def create_pii_sanitizer(detector: Optional[PIIDetector] = None) -> PIISanitizer:
    """Create a PII sanitizer instance."""
    return PIISanitizer(detector)


def sanitize_pii(text: str, 
                replacement_strategy: str = "hash",
                min_confidence: float = 0.5) -> str:
    """Convenience function to sanitize PII in text."""
    sanitizer = create_pii_sanitizer()
    result = sanitizer.sanitize_text(text, replacement_strategy, min_confidence=min_confidence)
    return result.sanitized_text