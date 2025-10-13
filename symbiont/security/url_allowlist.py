"""URL allowlist enforcement for notification integrations.

This module provides secure URL validation and allowlist enforcement
for external integrations like Slack, PagerDuty, etc.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class URLValidationError(Exception):
    """Raised when URL validation fails."""
    pass


class IntegrationType(Enum):
    """Types of external integrations."""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    EMAIL = "email"
    DISCORD = "discord"
    TEAMS = "teams"
    CUSTOM = "custom"


@dataclass
class URLRule:
    """Represents a URL allowlist rule."""
    pattern: str
    integration_type: IntegrationType
    description: str
    allowed_methods: List[str]
    required_headers: Optional[Dict[str, str]] = None
    max_redirects: int = 5
    timeout_seconds: int = 30


class URLAllowlist:
    """Manages URL allowlist rules and validation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules: List[URLRule] = []
        self._load_default_rules()
        self._load_custom_rules()
    
    def _load_default_rules(self) -> None:
        """Load default allowlist rules for common integrations."""
        default_rules = [
            # Slack
            URLRule(
                pattern=r"^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+$",
                integration_type=IntegrationType.SLACK,
                description="Slack webhook URLs",
                allowed_methods=["POST"],
                required_headers={"Content-Type": "application/json"}
            ),
            URLRule(
                pattern=r"^https://[a-zA-Z0-9-]+\.slack\.com/api/chat\.postMessage$",
                integration_type=IntegrationType.SLACK,
                description="Slack API chat.postMessage endpoint",
                allowed_methods=["POST"],
                required_headers={"Authorization": "Bearer .+"}
            ),
            
            # PagerDuty
            URLRule(
                pattern=r"^https://events\.pagerduty\.com/v2/enqueue$",
                integration_type=IntegrationType.PAGERDUTY,
                description="PagerDuty Events API v2",
                allowed_methods=["POST"],
                required_headers={"Content-Type": "application/json"}
            ),
            URLRule(
                pattern=r"^https://[a-zA-Z0-9-]+\.pagerduty\.com/api/v1/incidents$",
                integration_type=IntegrationType.PAGERDUTY,
                description="PagerDuty Incidents API",
                allowed_methods=["POST", "PUT"],
                required_headers={"Authorization": "Token token=.+"}
            ),
            
            # Discord
            URLRule(
                pattern=r"^https://discord\.com/api/webhooks/\d+/[A-Za-z0-9_-]+$",
                integration_type=IntegrationType.DISCORD,
                description="Discord webhook URLs",
                allowed_methods=["POST"],
                required_headers={"Content-Type": "application/json"}
            ),
            
            # Microsoft Teams
            URLRule(
                pattern=r"^https://[a-zA-Z0-9-]+\.webhook\.office\.com/webhookb2/[A-Za-z0-9-]+/[A-Za-z0-9-]+/[A-Za-z0-9-]+$",
                integration_type=IntegrationType.TEAMS,
                description="Microsoft Teams webhook URLs",
                allowed_methods=["POST"],
                required_headers={"Content-Type": "application/json"}
            ),
            
            # Generic webhooks (more restrictive)
            URLRule(
                pattern=r"^https://[a-zA-Z0-9.-]+\.webhook\.app/webhook/[A-Za-z0-9_-]+$",
                integration_type=IntegrationType.WEBHOOK,
                description="Generic webhook.app URLs",
                allowed_methods=["POST"],
                required_headers={"Content-Type": "application/json"}
            ),
        ]
        
        self.rules.extend(default_rules)
    
    def _load_custom_rules(self) -> None:
        """Load custom allowlist rules from config."""
        custom_rules = self.config.get("url_allowlist", {}).get("rules", [])
        
        for rule_config in custom_rules:
            try:
                rule = URLRule(
                    pattern=rule_config["pattern"],
                    integration_type=IntegrationType(rule_config["integration_type"]),
                    description=rule_config["description"],
                    allowed_methods=rule_config["allowed_methods"],
                    required_headers=rule_config.get("required_headers"),
                    max_redirects=rule_config.get("max_redirects", 5),
                    timeout_seconds=rule_config.get("timeout_seconds", 30)
                )
                self.rules.append(rule)
            except Exception as e:
                print(f"Warning: Failed to load custom URL rule: {e}")
    
    def validate_url(
        self, 
        url: str, 
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        integration_type: Optional[IntegrationType] = None
    ) -> Tuple[bool, Optional[URLRule], str]:
        """Validate URL against allowlist rules."""
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(url)
            
            # Basic URL validation
            if not parsed_url.scheme or not parsed_url.netloc:
                return False, None, "Invalid URL format"
            
            if parsed_url.scheme not in ["https", "http"]:
                return False, None, "Only HTTP and HTTPS URLs are allowed"
            
            # Check against rules
            for rule in self.rules:
                if integration_type and rule.integration_type != integration_type:
                    continue
                
                if re.match(rule.pattern, url):
                    # Check method
                    if method.upper() not in [m.upper() for m in rule.allowed_methods]:
                        return False, rule, f"Method {method} not allowed for this URL"
                    
                    # Check required headers
                    if rule.required_headers and headers:
                        for header_name, header_pattern in rule.required_headers.items():
                            header_value = headers.get(header_name, "")
                            if not re.match(header_pattern, header_value):
                                return False, rule, f"Required header {header_name} does not match pattern"
                    
                    return True, rule, "URL is allowed"
            
            return False, None, "URL not in allowlist"
            
        except Exception as e:
            return False, None, f"URL validation error: {e}"
    
    def add_rule(self, rule: URLRule) -> None:
        """Add a new allowlist rule."""
        self.rules.append(rule)
    
    def remove_rule(self, pattern: str, integration_type: IntegrationType) -> bool:
        """Remove an allowlist rule."""
        for i, rule in enumerate(self.rules):
            if rule.pattern == pattern and rule.integration_type == integration_type:
                del self.rules[i]
                return True
        return False
    
    def list_rules(self) -> List[URLRule]:
        """List all allowlist rules."""
        return self.rules.copy()
    
    def get_rules_for_integration(self, integration_type: IntegrationType) -> List[URLRule]:
        """Get rules for a specific integration type."""
        return [rule for rule in self.rules if rule.integration_type == integration_type]


class URLValidator:
    """Validates URLs for security and compliance."""
    
    def __init__(self, allowlist: URLAllowlist):
        self.allowlist = allowlist
    
    def validate_notification_url(
        self,
        url: str,
        integration_type: IntegrationType,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Validate a notification URL for security."""
        is_allowed, rule, message = self.allowlist.validate_url(
            url, method, headers, integration_type
        )
        
        # Additional security checks
        security_issues = self._check_security_issues(url)
        
        return {
            "is_allowed": is_allowed,
            "rule": rule,
            "message": message,
            "security_issues": security_issues,
            "recommendations": self._get_recommendations(url, security_issues)
        }
    
    def _check_security_issues(self, url: str) -> List[str]:
        """Check for security issues in URL."""
        issues = []
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check for HTTP (non-HTTPS)
            if parsed.scheme == "http":
                issues.append("URL uses HTTP instead of HTTPS")
            
            # Check for localhost/internal IPs
            if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                issues.append("URL points to localhost")
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r"\.local$",
                r"\.internal$",
                r"\.test$",
                r"\.dev$",
                r"\.example$"
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, parsed.hostname or ""):
                    issues.append(f"URL hostname matches suspicious pattern: {pattern}")
            
            # Check for IP addresses
            ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
            if re.match(ip_pattern, parsed.hostname or ""):
                issues.append("URL uses IP address instead of domain name")
            
            # Check for suspicious ports
            if parsed.port and parsed.port not in [80, 443, 8080, 8443]:
                issues.append(f"URL uses non-standard port: {parsed.port}")
            
        except Exception as e:
            issues.append(f"Error parsing URL: {e}")
        
        return issues
    
    def _get_recommendations(self, url: str, security_issues: List[str]) -> List[str]:
        """Get security recommendations for URL."""
        recommendations = []
        
        if not security_issues:
            recommendations.append("URL appears secure")
            return recommendations
        
        if "URL uses HTTP instead of HTTPS" in security_issues:
            recommendations.append("Use HTTPS instead of HTTP")
        
        if "URL points to localhost" in security_issues:
            recommendations.append("Use a proper domain name instead of localhost")
        
        if "URL uses IP address instead of domain name" in security_issues:
            recommendations.append("Use a domain name instead of IP address")
        
        if "URL uses non-standard port" in security_issues:
            recommendations.append("Consider using standard ports (80, 443)")
        
        return recommendations


class NotificationSecurityManager:
    """Manages security for notification integrations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.allowlist = URLAllowlist(config)
        self.validator = URLValidator(self.allowlist)
        self.blocked_urls: Set[str] = set()
        self.allowed_domains: Set[str] = set()
        self._load_security_config()
    
    def _load_security_config(self) -> None:
        """Load security configuration."""
        config = self.allowlist.config
        self.blocked_urls = set(config.get("url_allowlist", {}).get("blocked_urls", []))
        self.allowed_domains = set(config.get("url_allowlist", {}).get("allowed_domains", []))
    
    def is_url_allowed(
        self,
        url: str,
        integration_type: IntegrationType,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Check if URL is allowed for notification."""
        # Check blocked URLs first
        if url in self.blocked_urls:
            return False
        
        # Check domain allowlist if configured
        if self.allowed_domains:
            try:
                parsed = urllib.parse.urlparse(url)
                domain = parsed.hostname
                if domain and not any(domain.endswith(allowed) for allowed in self.allowed_domains):
                    return False
            except Exception:
                return False
        
        # Check against allowlist rules
        is_allowed, _, _ = self.allowlist.validate_url(url, method, headers, integration_type)
        return is_allowed
    
    def validate_notification_request(
        self,
        url: str,
        integration_type: IntegrationType,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a complete notification request."""
        validation_result = self.validator.validate_notification_url(
            url, integration_type, method, headers
        )
        
        # Additional payload validation
        payload_issues = []
        if payload:
            payload_issues = self._validate_payload(payload, integration_type)
        
        return {
            **validation_result,
            "payload_issues": payload_issues,
            "is_safe": validation_result["is_allowed"] and not payload_issues
        }
    
    def _validate_payload(self, payload: Dict[str, Any], integration_type: IntegrationType) -> List[str]:
        """Validate notification payload for security."""
        issues = []
        
        # Check for sensitive data in payload
        sensitive_patterns = [
            r"(?i)password",
            r"(?i)secret",
            r"(?i)key",
            r"(?i)token",
            r"(?i)auth",
            r"(?i)credential"
        ]
        
        payload_str = str(payload).lower()
        for pattern in sensitive_patterns:
            if re.search(pattern, payload_str):
                issues.append(f"Payload may contain sensitive data matching pattern: {pattern}")
        
        # Integration-specific validation
        if integration_type == IntegrationType.SLACK:
            if "text" in payload and len(payload["text"]) > 4000:
                issues.append("Slack message text too long (max 4000 characters)")
        
        elif integration_type == IntegrationType.DISCORD:
            if "content" in payload and len(payload["content"]) > 2000:
                issues.append("Discord message content too long (max 2000 characters)")
        
        return issues
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security report for notification system."""
        return {
            "total_rules": len(self.allowlist.rules),
            "blocked_urls": len(self.blocked_urls),
            "allowed_domains": len(self.allowed_domains),
            "rules_by_integration": {
                integration_type.value: len(self.allowlist.get_rules_for_integration(integration_type))
                for integration_type in IntegrationType
            }
        }


# Convenience functions
def validate_slack_webhook(url: str) -> Dict[str, Any]:
    """Validate Slack webhook URL."""
    manager = NotificationSecurityManager()
    return manager.validate_notification_request(
        url, IntegrationType.SLACK, "POST"
    )


def validate_pagerduty_webhook(url: str) -> Dict[str, Any]:
    """Validate PagerDuty webhook URL."""
    manager = NotificationSecurityManager()
    return manager.validate_notification_request(
        url, IntegrationType.PAGERDUTY, "POST"
    )


def validate_discord_webhook(url: str) -> Dict[str, Any]:
    """Validate Discord webhook URL."""
    manager = NotificationSecurityManager()
    return manager.validate_notification_request(
        url, IntegrationType.DISCORD, "POST"
    )