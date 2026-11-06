"""How text is prepared for embedding.

Kept in one place so that indexed papers and submitted manuscripts are always prepared the
same way, and so that the exact rule is easy to find, test and explain.
"""

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (arXiv abstracts contain hard line breaks) into one space."""
    return _WHITESPACE.sub(" ", text).strip()


def build_embedding_text(title: str | None, body: str) -> str:
    """Build the text that gets embedded: ``"<title>. <body>"``, or just the body.

    The same rule is used for arXiv papers (body = abstract) and for submissions
    (body = manuscript text), so both sides of a comparison are built the same way.
    """
    clean_body = normalize_whitespace(body)
    clean_title = normalize_whitespace(title) if title else ""
    if not clean_title:
        return clean_body
    separator = " " if clean_title.endswith((".", "?", "!")) else ". "
    return f"{clean_title}{separator}{clean_body}"


def content_hash(text: str) -> str:
    """Stable fingerprint of the embedded text, used to skip re-embedding unchanged papers."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
