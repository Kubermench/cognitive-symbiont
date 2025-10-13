"""URL allowlist enforcement for notification integrations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse, urljoin
import ipaddress

logger = logging.getLogger(__name__)


class URLPolicy(Enum):
    """URL policy types."""
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


@dataclass
class URLRule:
    """A URL allowlist rule."""
    pattern: str
    policy: URLPolicy
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class URLValidationResult:
    """Result of URL validation."""
    url: str
    allowed: bool
    policy: URLPolicy
    matched_rule: Optional[URLRule] = None
    reason: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class URLAllowlist:
    """URL allowlist manager with pattern matching and validation."""
    
    def __init__(self):
        self.rules: List[URLRule] = []
        self.default_policy = URLPolicy.DENY
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    def add_rule(self, pattern: str, policy: URLPolicy, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a URL rule to the allowlist."""
        try:
            # Validate pattern
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self._compiled_patterns[pattern] = compiled_pattern
            
            rule = URLRule(
                pattern=pattern,
                policy=policy,
                description=description,
                metadata=metadata or {}
            )
            
            self.rules.append(rule)
            logger.info("Added URL rule: %s -> %s", pattern, policy.value)
            return True
            
        except re.error as e:
            logger.error("Invalid regex pattern '%s': %s", pattern, e)
            return False
    
    def remove_rule(self, pattern: str) -> bool:
        """Remove a URL rule from the allowlist."""
        for i, rule in enumerate(self.rules):
            if rule.pattern == pattern:
                del self.rules[i]
                if pattern in self._compiled_patterns:
                    del self._compiled_patterns[pattern]
                logger.info("Removed URL rule: %s", pattern)
                return True
        return False
    
    def validate_url(self, url: str, context: Optional[Dict[str, Any]] = None) -> URLValidationResult:
        """Validate a URL against the allowlist."""
        if not url:
            return URLValidationResult(
                url=url,
                allowed=False,
                policy=URLPolicy.DENY,
                reason="Empty URL"
            )
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return URLValidationResult(
                url=url,
                allowed=False,
                policy=URLPolicy.DENY,
                reason=f"Invalid URL format: {e}"
            )
        
        # Check if URL is valid
        if not parsed.scheme or not parsed.netloc:
            return URLValidationResult(
                url=url,
                allowed=False,
                policy=URLPolicy.DENY,
                reason="Invalid URL: missing scheme or netloc"
            )
        
        # Check against rules (in order)
        for rule in self.rules:
            if self._matches_rule(url, rule):
                return URLValidationResult(
                    url=url,
                    allowed=rule.policy == URLPolicy.ALLOW,
                    policy=rule.policy,
                    matched_rule=rule,
                    reason=f"Matched rule: {rule.description or rule.pattern}"
                )
        
        # No rules matched, use default policy
        return URLValidationResult(
            url=url,
            allowed=self.default_policy == URLPolicy.ALLOW,
            policy=self.default_policy,
            reason="No matching rule found, using default policy"
        )
    
    def _matches_rule(self, url: str, rule: URLRule) -> bool:
        """Check if a URL matches a rule."""
        try:
            pattern = self._compiled_patterns.get(rule.pattern)
            if not pattern:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                self._compiled_patterns[rule.pattern] = pattern
            
            return bool(pattern.search(url))
            
        except Exception as e:
            logger.warning("Error matching URL '%s' against pattern '%s': %s", url, rule.pattern, e)
            return False
    
    def get_allowed_domains(self) -> Set[str]:
        """Get all allowed domains from rules."""
        domains = set()
        
        for rule in self.rules:
            if rule.policy == URLPolicy.ALLOW:
                try:
                    # Extract domain from pattern
                    if rule.pattern.startswith('https?://'):
                        # Pattern starts with protocol
                        match = re.search(r'https?://([^/]+)', rule.pattern)
                        if match:
                            domain = match.group(1)
                            # Remove port and path
                            domain = domain.split(':')[0].split('/')[0]
                            domains.add(domain)
                    elif not rule.pattern.startswith('^'):
                        # Simple domain pattern
                        domain = rule.pattern.split('/')[0].split(':')[0]
                        if '.' in domain and not domain.startswith('.'):
                            domains.add(domain)
                except Exception:
                    continue
        
        return domains
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export rules as dictionary list."""
        return [
            {
                "pattern": rule.pattern,
                "policy": rule.policy.value,
                "description": rule.description,
                "metadata": rule.metadata
            }
            for rule in self.rules
        ]
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """Import rules from dictionary list."""
        imported = 0
        
        for rule_data in rules_data:
            try:
                pattern = rule_data["pattern"]
                policy = URLPolicy(rule_data["policy"])
                description = rule_data.get("description", "")
                metadata = rule_data.get("metadata", {})
                
                if self.add_rule(pattern, policy, description, metadata):
                    imported += 1
                    
            except Exception as e:
                logger.warning("Failed to import rule %s: %s", rule_data, e)
        
        return imported


class NotificationURLValidator:
    """URL validator for notification integrations."""
    
    def __init__(self, allowlist: Optional[URLAllowlist] = None):
        self.allowlist = allowlist or URLAllowlist()
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default safe URL rules."""
        # Allow common notification services
        default_rules = [
            # Slack
            ("https://hooks.slack.com/services/.*", URLPolicy.ALLOW, "Slack webhooks"),
            ("https://api.slack.com/.*", URLPolicy.ALLOW, "Slack API"),
            
            # Discord
            ("https://discord.com/api/webhooks/.*", URLPolicy.ALLOW, "Discord webhooks"),
            ("https://discordapp.com/api/webhooks/.*", URLPolicy.ALLOW, "Discord webhooks (legacy)"),
            
            # Microsoft Teams
            ("https://.*\.webhook\.office\.com/.*", URLPolicy.ALLOW, "Microsoft Teams webhooks"),
            
            # PagerDuty
            ("https://events.pagerduty.com/.*", URLPolicy.ALLOW, "PagerDuty events"),
            ("https://api.pagerduty.com/.*", URLPolicy.ALLOW, "PagerDuty API"),
            
            # GitHub
            ("https://api.github.com/.*", URLPolicy.ALLOW, "GitHub API"),
            ("https://github.com/.*", URLPolicy.ALLOW, "GitHub"),
            
            # GitLab
            ("https://gitlab.com/.*", URLPolicy.ALLOW, "GitLab"),
            ("https://.*\.gitlab\.com/.*", URLPolicy.ALLOW, "GitLab instances"),
            
            # Jira
            ("https://.*\.atlassian\.net/.*", URLPolicy.ALLOW, "Atlassian/Jira"),
            
            # Generic webhook patterns (more restrictive)
            ("https://.*\.webhook\.app/.*", URLPolicy.ALLOW, "Generic webhook apps"),
            ("https://webhook\.site/.*", URLPolicy.ALLOW, "Webhook.site"),
            
            # Local development
            ("http://localhost:.*", URLPolicy.ALLOW, "Local development"),
            ("http://127\.0\.0\.1:.*", URLPolicy.ALLOW, "Local development"),
            ("https://localhost:.*", URLPolicy.ALLOW, "Local development (HTTPS)"),
            
            # Deny dangerous patterns
            ("file://.*", URLPolicy.DENY, "File URLs"),
            ("ftp://.*", URLPolicy.DENY, "FTP URLs"),
            ("javascript:.*", URLPolicy.DENY, "JavaScript URLs"),
            ("data:.*", URLPolicy.DENY, "Data URLs"),
            (".*@.*", URLPolicy.DENY, "Email addresses in URLs"),
        ]
        
        for pattern, policy, description in default_rules:
            self.allowlist.add_rule(pattern, policy, description)
    
    def validate_webhook_url(self, url: str, service: Optional[str] = None) -> URLValidationResult:
        """Validate a webhook URL."""
        result = self.allowlist.validate_url(url)
        
        # Add service-specific warnings
        if result.allowed and service:
            warnings = self._get_service_warnings(url, service)
            result.warnings.extend(warnings)
        
        return result
    
    def _get_service_warnings(self, url: str, service: str) -> List[str]:
        """Get service-specific warnings for a URL."""
        warnings = []
        
        if service.lower() == "slack":
            if not url.startswith("https://hooks.slack.com/services/"):
                warnings.append("URL does not match standard Slack webhook format")
        
        elif service.lower() == "discord":
            if not ("discord.com/api/webhooks/" in url or "discordapp.com/api/webhooks/" in url):
                warnings.append("URL does not match standard Discord webhook format")
        
        elif service.lower() == "teams":
            if not ".webhook.office.com" in url:
                warnings.append("URL does not match standard Teams webhook format")
        
        # Check for HTTPS
        if not url.startswith("https://"):
            warnings.append("URL should use HTTPS for security")
        
        # Check for suspicious patterns
        if any(suspicious in url.lower() for suspicious in ["admin", "root", "config", "secret"]):
            warnings.append("URL contains potentially sensitive keywords")
        
        return warnings
    
    def validate_notification_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate notification configuration for URL safety."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validated_urls": {}
        }
        
        # Check common URL fields
        url_fields = ["url", "webhook_url", "endpoint", "callback_url", "notify_url"]
        
        for field in url_fields:
            if field in config and config[field]:
                url = config[field]
                validation = self.validate_webhook_url(url, config.get("service"))
                
                results["validated_urls"][field] = {
                    "url": url,
                    "allowed": validation.allowed,
                    "policy": validation.policy.value,
                    "reason": validation.reason,
                    "warnings": validation.warnings
                }
                
                if not validation.allowed:
                    results["valid"] = False
                    results["errors"].append(f"{field}: {validation.reason}")
                
                results["warnings"].extend(validation.warnings)
        
        return results


def create_url_allowlist() -> URLAllowlist:
    """Create a URL allowlist instance."""
    return URLAllowlist()


def create_notification_validator(allowlist: Optional[URLAllowlist] = None) -> NotificationURLValidator:
    """Create a notification URL validator."""
    return NotificationURLValidator(allowlist)


def validate_webhook_url(url: str, service: Optional[str] = None) -> URLValidationResult:
    """Convenience function to validate a webhook URL."""
    validator = create_notification_validator()
    return validator.validate_webhook_url(url, service)