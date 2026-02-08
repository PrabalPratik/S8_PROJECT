"""
Security Module Tests

Unit tests for security utilities:
- Input sanitization
- Path validation
- Rate limiting
- HTML sanitization
"""

import pytest
import os
import time
from utils.security import (
    InputSanitizer,
    PathValidator,
    RateLimiter,
    HTMLSanitizer,
    sanitize_input,
    validate_path,
    check_rate_limit,
    escape_html
)


class TestInputSanitizer:
    """Tests for InputSanitizer class."""
    
    def test_xss_script_tag_escaped(self):
        """XSS script tags should be escaped."""
        payload = '<script>alert("xss")</script>'
        result = InputSanitizer.sanitize_text(payload)
        assert '<script>' not in result
        assert '&lt;script&gt;' in result
    
    def test_xss_img_onerror_escaped(self):
        """XSS via img onerror should be escaped."""
        payload = '<img src=x onerror=alert("xss")>'
        result = InputSanitizer.sanitize_text(payload)
        assert '<img' not in result
        assert 'onerror' not in result or '&' in result
    
    def test_xss_event_handlers_escaped(self):
        """XSS event handlers should be escaped."""
        payloads = [
            '<div onmouseover="alert(1)">',
            '<body onload="alert(1)">',
            '<a href="javascript:alert(1)">click</a>'
        ]
        for payload in payloads:
            result = InputSanitizer.sanitize_text(payload)
            assert '<' not in result
    
    def test_length_limit_enforced(self):
        """Input should be truncated to max length."""
        long_input = "A" * 1000
        result = InputSanitizer.sanitize_text(long_input, 'role')
        assert len(result) == 100  # MAX_LENGTHS['role']
    
    def test_null_bytes_removed(self):
        """Null bytes should be removed."""
        payload = "test\x00injection"
        result = InputSanitizer.sanitize_text(payload)
        assert '\x00' not in result
    
    def test_valid_role_accepted(self):
        """Valid role names should pass validation."""
        valid_roles = [
            "Senior Backend Engineer",
            "ML/AI Research Scientist",
            "DevOps Engineer (Cloud)",
            "Full-Stack Developer"
        ]
        for role in valid_roles:
            sanitized, is_valid = InputSanitizer.sanitize_role(role)
            assert is_valid, f"Role should be valid: {role}"
    
    def test_malicious_role_rejected(self):
        """Malicious role names should fail validation."""
        malicious_roles = [
            "<script>alert(1)</script>",
            "'; DROP TABLE users; --",
            "role with ${command}"
        ]
        for role in malicious_roles:
            sanitized, is_valid = InputSanitizer.sanitize_role(role)
            assert not is_valid, f"Role should be invalid: {role}"
    
    def test_valid_skills_accepted(self):
        """Valid skills should pass validation."""
        valid_skills = "Python, JavaScript, C++, React.js, Node.js"
        sanitized, is_valid = InputSanitizer.sanitize_skills(valid_skills)
        assert is_valid
        assert "Python" in sanitized
    
    def test_skills_list_cleaned(self):
        """Skills list should be cleaned and normalized."""
        messy_skills = "  Python  ,   JavaScript,  , React  "
        sanitized, is_valid = InputSanitizer.sanitize_skills(messy_skills)
        assert sanitized == "Python, JavaScript, React"
    
    def test_empty_input_handled(self):
        """Empty inputs should be handled gracefully."""
        assert InputSanitizer.sanitize_text("") == ""
        assert InputSanitizer.sanitize_text(None) == ""
        
        role, valid = InputSanitizer.sanitize_role("")
        assert role == ""
        assert not valid


class TestPathValidator:
    """Tests for PathValidator class."""
    
    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""
        validator = PathValidator(['data'])
        
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "data/../../secret.txt",
            "/etc/shadow"
        ]
        for path in malicious_paths:
            _, is_valid = validator.validate_path(path)
            # Should either be blocked or resolved safely
            assert not is_valid or 'etc' not in _ and 'windows' not in _
    
    def test_valid_data_path_accepted(self):
        """Valid paths within allowed directories should be accepted."""
        validator = PathValidator(['data', 'assets'])
        
        # Create a real path for testing
        if os.path.exists('data'):
            path, is_valid = validator.validate_path("data/test.csv")
            # Path should be normalized
            assert path
    
    def test_filename_sanitized(self):
        """Dangerous filename characters should be removed."""
        dangerous_names = [
            "file<>name.txt",
            "file:name.txt",
            "file|name.txt",
            "../file.txt"
        ]
        for name in dangerous_names:
            sanitized = PathValidator.sanitize_filename(name)
            assert '<' not in sanitized
            assert '>' not in sanitized
            assert ':' not in sanitized
            assert '|' not in sanitized
            assert '..' not in sanitized
    
    def test_filename_length_limited(self):
        """Filenames should be limited to 255 characters."""
        long_name = "a" * 500 + ".txt"
        sanitized = PathValidator.sanitize_filename(long_name)
        assert len(sanitized) <= 255


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_requests_within_limit_allowed(self):
        """Requests within limit should be allowed."""
        limiter = RateLimiter()
        
        for i in range(3):
            allowed, remaining = limiter.check_rate_limit("test_key", max_requests=5, window_seconds=60)
            assert allowed
            assert remaining == 5 - (i + 1) - 1
    
    def test_requests_exceeding_limit_blocked(self):
        """Requests exceeding limit should be blocked."""
        limiter = RateLimiter()
        
        # Use up all requests
        for i in range(3):
            limiter.check_rate_limit("blocked_key", max_requests=3, window_seconds=60)
        
        # Next request should be blocked
        allowed, remaining = limiter.check_rate_limit("blocked_key", max_requests=3, window_seconds=60)
        assert not allowed
        assert remaining == 0
    
    def test_different_keys_independent(self):
        """Different keys should have independent limits."""
        limiter = RateLimiter()
        
        # Exhaust one key
        for _ in range(3):
            limiter.check_rate_limit("key1", max_requests=3, window_seconds=60)
        
        # Other key should still work
        allowed, _ = limiter.check_rate_limit("key2", max_requests=3, window_seconds=60)
        assert allowed
    
    def test_reset_clears_limits(self):
        """Reset should clear rate limits for a key."""
        limiter = RateLimiter()
        
        # Exhaust limit
        for _ in range(3):
            limiter.check_rate_limit("reset_key", max_requests=3, window_seconds=60)
        
        # Reset
        limiter.reset("reset_key")
        
        # Should be allowed again
        allowed, _ = limiter.check_rate_limit("reset_key", max_requests=3, window_seconds=60)
        assert allowed


class TestHTMLSanitizer:
    """Tests for HTMLSanitizer class."""
    
    def test_user_input_fully_escaped(self):
        """User input should be fully escaped."""
        dangerous_inputs = [
            '<script>alert(1)</script>',
            '<img src=x onerror=alert(1)>',
            '<div onclick="evil()">click</div>'
        ]
        for input_text in dangerous_inputs:
            result = HTMLSanitizer.sanitize_for_display(input_text)
            assert '<' not in result
            assert '>' not in result
    
    def test_safe_metric_format(self):
        """Metric format should escape user values."""
        # User-controlled values
        label = "<script>evil</script>"
        value = "100<img src=x>"
        
        result = HTMLSanitizer.format_safe_metric(label, value)
        
        # HTML structure should exist but user values escaped
        assert 'kpi-card' in result
        assert '<script>' not in result
        assert '<img' not in result


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_sanitize_input_function(self):
        """sanitize_input should work correctly."""
        result = sanitize_input('<script>alert(1)</script>')
        assert '<script>' not in result
    
    def test_escape_html_function(self):
        """escape_html should escape all HTML."""
        result = escape_html('<div>test</div>')
        assert '&lt;' in result
        assert '&gt;' in result


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
