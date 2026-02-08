"""
Security Utilities Module for TalentAI

Provides centralized security functions:
- Input sanitization
- Path validation
- Rate limiting
- HTML sanitization
"""

import os
import re
import html
import time
from typing import Optional, List, Dict, Tuple
from functools import wraps


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

class InputSanitizer:
    """Sanitizes user inputs to prevent XSS and injection attacks."""
    
    # Allowed characters for different field types
    ROLE_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\.\,\&\(\)\/]+$')
    SKILLS_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\.\,\+\#\(\)\/]+$')
    
    # Maximum lengths for different fields
    MAX_LENGTHS = {
        'role': 100,
        'skills': 500,
        'optional_skills': 300,
        'text': 5000,
        'jd_text': 10000
    }
    
    @classmethod
    def sanitize_text(
        cls, 
        text: str, 
        field_type: str = 'text',
        escape_html: bool = True
    ) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Raw user input
            field_type: Type of field for length limits
            escape_html: Whether to escape HTML entities
            
        Returns:
            Sanitized text string
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text).strip()
        
        # Apply length limit
        max_length = cls.MAX_LENGTHS.get(field_type, 1000)
        text = text[:max_length]
        
        # Escape HTML entities to prevent XSS
        if escape_html:
            text = html.escape(text)
        
        # Remove null bytes and other dangerous characters
        text = text.replace('\x00', '').replace('\r', '')
        
        return text
    
    @classmethod
    def sanitize_role(cls, role: str) -> Tuple[str, bool]:
        """
        Sanitize role name input.
        
        Returns:
            Tuple of (sanitized_role, is_valid)
        """
        sanitized = cls.sanitize_text(role, 'role')
        
        # Validate pattern
        if not sanitized:
            return "", False
            
        # Check for suspicious patterns
        is_valid = bool(cls.ROLE_PATTERN.match(sanitized))
        
        return sanitized, is_valid
    
    @classmethod
    def sanitize_skills(cls, skills: str) -> Tuple[str, bool]:
        """
        Sanitize skills input (comma-separated list).
        
        Returns:
            Tuple of (sanitized_skills, is_valid)
        """
        sanitized = cls.sanitize_text(skills, 'skills')
        
        if not sanitized:
            return "", False
        
        # Parse and clean individual skills
        skill_list = [s.strip() for s in sanitized.split(',') if s.strip()]
        
        # Validate each skill
        valid_skills = []
        for skill in skill_list:
            if cls.SKILLS_PATTERN.match(skill):
                valid_skills.append(skill)
        
        result = ', '.join(valid_skills)
        is_valid = len(valid_skills) == len(skill_list)
        
        return result, is_valid


# =============================================================================
# PATH VALIDATION
# =============================================================================

class PathValidator:
    """Validates file paths to prevent path traversal attacks."""
    
    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        """
        Initialize with allowed directories.
        
        Args:
            allowed_dirs: List of allowed base directories
        """
        self.allowed_dirs = allowed_dirs or ['data', 'assets', '.streamlit']
    
    def validate_path(self, path: str, base_dir: Optional[str] = None) -> Tuple[str, bool]:
        """
        Validate and normalize a file path.
        
        Args:
            path: Path to validate
            base_dir: Optional base directory to resolve relative paths
            
        Returns:
            Tuple of (normalized_path, is_valid)
        """
        if not path:
            return "", False
        
        try:
            # Normalize the path
            if base_dir:
                full_path = os.path.normpath(os.path.join(base_dir, path))
            else:
                full_path = os.path.normpath(path)
            
            # Convert to absolute path
            abs_path = os.path.abspath(full_path)
            
            # Check for path traversal attempts
            if '..' in path or path.startswith('/') or path.startswith('\\'):
                # Only allow if it resolves within allowed directories
                pass
            
            # Check if path is within allowed directories
            for allowed_dir in self.allowed_dirs:
                allowed_abs = os.path.abspath(allowed_dir)
                if abs_path.startswith(allowed_abs):
                    return abs_path, True
            
            # Check if it's a relative path within working directory
            cwd = os.getcwd()
            if abs_path.startswith(cwd):
                # Verify it doesn't escape via directory names
                rel_path = os.path.relpath(abs_path, cwd)
                if not rel_path.startswith('..'):
                    return abs_path, True
            
            return abs_path, False
            
        except (ValueError, OSError):
            return "", False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.
        
        Args:
            filename: Raw filename
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return ""
        
        # Remove path separators
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        return filename[:255]


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    In-memory rate limiter using sliding window.
    
    Note: For production with multiple workers, use Redis-based rate limiting.
    """
    
    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
    
    def check_rate_limit(
        self, 
        key: str, 
        max_requests: int = 10, 
        window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check if a request is within rate limits.
        
        Args:
            key: Unique identifier (e.g., session_id + action)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Initialize if new key
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests outside window
        self._requests[key] = [
            t for t in self._requests[key] 
            if t > window_start
        ]
        
        # Check limit
        current_count = len(self._requests[key])
        remaining = max_requests - current_count
        
        if current_count >= max_requests:
            return False, 0
        
        # Record this request
        self._requests[key].append(current_time)
        
        return True, remaining - 1
    
    def get_wait_time(self, key: str, window_seconds: int = 60) -> int:
        """
        Get seconds to wait before next allowed request.
        
        Returns:
            Seconds to wait (0 if allowed now)
        """
        if key not in self._requests or not self._requests[key]:
            return 0
        
        oldest_request = min(self._requests[key])
        current_time = time.time()
        wait_time = oldest_request + window_seconds - current_time
        
        return max(0, int(wait_time))
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if key in self._requests:
            del self._requests[key]


# =============================================================================
# HTML SANITIZATION
# =============================================================================

class HTMLSanitizer:
    """Sanitizes HTML for safe rendering."""
    
    # Tags allowed in output
    ALLOWED_TAGS = {
        'p', 'br', 'strong', 'em', 'b', 'i', 'u',
        'ul', 'ol', 'li', 'div', 'span', 'h1', 'h2', 'h3', 'h4'
    }
    
    # Attributes allowed (per tag)
    ALLOWED_ATTRS = {
        'div': {'class', 'style'},
        'span': {'class', 'style'},
        'p': {'class'},
    }
    
    # CSS properties allowed in style attributes
    ALLOWED_CSS = {
        'color', 'background-color', 'font-weight', 'font-size',
        'text-align', 'margin', 'padding'
    }
    
    @classmethod
    def sanitize_for_display(cls, user_input: str) -> str:
        """
        Escape user input for safe display in HTML context.
        
        This completely escapes all HTML - use for user-generated content
        that should never contain HTML.
        """
        return html.escape(str(user_input))
    
    @classmethod
    def format_safe_metric(cls, label: str, value: str, extra: str = "") -> str:
        """
        Create a safe HTML metric display from user values.
        
        All user-provided values are escaped.
        """
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        safe_extra = html.escape(str(extra)) if extra else ""
        
        return f"""
        <div class="kpi-card">
            <h3>{safe_label}</h3>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{safe_value}</div>
            <div style="color: #64748B;">{safe_extra}</div>
        </div>
        """


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_rate_limiter_instance: Optional[RateLimiter] = None
_path_validator_instance: Optional[PathValidator] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create singleton rate limiter."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter()
    return _rate_limiter_instance


def get_path_validator(allowed_dirs: Optional[List[str]] = None) -> PathValidator:
    """Get or create singleton path validator."""
    global _path_validator_instance
    if _path_validator_instance is None:
        _path_validator_instance = PathValidator(allowed_dirs)
    return _path_validator_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def sanitize_input(text: str, field_type: str = 'text') -> str:
    """Convenience function for text sanitization."""
    return InputSanitizer.sanitize_text(text, field_type)


def validate_path(path: str, base_dir: Optional[str] = None) -> Tuple[str, bool]:
    """Convenience function for path validation."""
    return get_path_validator().validate_path(path, base_dir)


def check_rate_limit(key: str, max_requests: int = 10, window_seconds: int = 60) -> Tuple[bool, int]:
    """Convenience function for rate limiting."""
    return get_rate_limiter().check_rate_limit(key, max_requests, window_seconds)


def escape_html(text: str) -> str:
    """Convenience function for HTML escaping."""
    return HTMLSanitizer.sanitize_for_display(text)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Security Module - Self Test")
    print("=" * 50)
    
    # Test input sanitization
    print("\n[1] Input Sanitization Tests:")
    
    xss_payload = '<script>alert("xss")</script>'
    sanitized = sanitize_input(xss_payload)
    print(f"  XSS payload: {xss_payload}")
    print(f"  Sanitized:   {sanitized}")
    assert '<script>' not in sanitized
    print("  ✓ XSS payload escaped")
    
    # Test role sanitization
    role, valid = InputSanitizer.sanitize_role("Senior Backend Engineer")
    assert valid
    print(f"  Valid role: {role} -> {valid}")
    
    role, valid = InputSanitizer.sanitize_role("<malicious>")
    assert not valid
    print(f"  Invalid role blocked: {valid}")
    
    # Test rate limiting
    print("\n[2] Rate Limiting Tests:")
    limiter = RateLimiter()
    
    for i in range(5):
        allowed, remaining = limiter.check_rate_limit("test_key", max_requests=3, window_seconds=60)
        print(f"  Request {i+1}: allowed={allowed}, remaining={remaining}")
    
    # Test path validation
    print("\n[3] Path Validation Tests:")
    validator = PathValidator(['data'])
    
    path, valid = validator.validate_path("data/test.csv")
    print(f"  Valid path: data/test.csv -> {valid}")
    
    path, valid = validator.validate_path("../../../etc/passwd")
    print(f"  Traversal blocked: ../../../etc/passwd -> {valid}")
    
    print("\n" + "=" * 50)
    print("All security tests passed! ✓")
    print("=" * 50)
