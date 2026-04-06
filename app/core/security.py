"""
Security utilities for request validation.

Covers:
- SSRF prevention (block internal/private IPs in URL endpoints)
- Content-length validation for remote downloads
- URL scheme validation

Interview Points:
- SSRF is a top-10 OWASP vulnerability. Any endpoint that fetches
  user-supplied URLs MUST validate the target is not an internal resource.
- Private IP ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16,
  169.254.0.0/16 (AWS metadata), 127.0.0.0/8 (loopback)
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Maximum allowed download size (bytes) for URL-based image search
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_url(url: str) -> None:
    """
    Validate a URL for safety before making a request.

    Raises:
        ValueError: If the URL is unsafe (internal IP, bad scheme, etc.)
    """
    parsed = urlparse(url)

    # Scheme validation
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only HTTP/HTTPS URLs are allowed, got: {parsed.scheme}")

    # Hostname validation
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")

    # Block known dangerous hostnames
    blocked_hosts = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "metadata.google.internal",
    }
    if hostname.lower() in blocked_hosts:
        raise ValueError(f"Access to {hostname} is not allowed")

    # Resolve hostname and check for private IPs
    try:
        resolved_ip = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(resolved_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"URL resolves to private/internal IP ({resolved_ip}). "
                "Access denied for security reasons."
            )
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    logger.debug("URL validated: %s → %s", url, resolved_ip)


def check_content_length(headers: dict, max_bytes: int = MAX_DOWNLOAD_BYTES) -> None:
    """
    Validate Content-Length header before downloading.

    Args:
        headers: Response headers dict
        max_bytes: Maximum allowed content length

    Raises:
        ValueError: If content exceeds the limit
    """
    # Case-insensitive lookup (plain dicts lose httpx's case-insensitivity)
    content_length = None
    for key, value in headers.items():
        if key.lower() == "content-length":
            content_length = value
            break

    if content_length:
        try:
            size = int(content_length)
        except (ValueError, TypeError):
            logger.warning("Malformed Content-Length header: %s", content_length)
            return  # Can't validate — proceed with download
        if size > max_bytes:
            raise ValueError(
                f"Response too large: {size / 1024 / 1024:.1f}MB "
                f"(max: {max_bytes / 1024 / 1024:.0f}MB)"
            )
