"""Tests for API key encryption/decryption (opinionforge.storage.encryption).

All tests use temporary directories for key file storage to avoid
polluting the user's real application data directory.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from opinionforge.storage.encryption import (
    _get_or_create_key,
    decrypt_key,
    encrypt_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_key_path(tmp_path: Path) -> Path:
    """Return a temporary key file path inside *tmp_path*."""
    return tmp_path / "test_secret.key"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEncryptDecryptRoundTrip:
    """encrypt_key -> decrypt_key should return the original plaintext."""

    def test_round_trip_simple(self, tmp_path: Path) -> None:
        """Encrypt then decrypt returns original text."""
        kp = _tmp_key_path(tmp_path)
        original = "sk-abc123XYZ"
        encrypted = encrypt_key(original, key_path=kp)
        decrypted = decrypt_key(encrypted, key_path=kp)
        assert decrypted == original

    def test_round_trip_empty_string(self, tmp_path: Path) -> None:
        """Empty string encrypts and decrypts correctly."""
        kp = _tmp_key_path(tmp_path)
        encrypted = encrypt_key("", key_path=kp)
        decrypted = decrypt_key(encrypted, key_path=kp)
        assert decrypted == ""

    def test_round_trip_unicode(self, tmp_path: Path) -> None:
        """Unicode characters survive the encrypt/decrypt cycle."""
        kp = _tmp_key_path(tmp_path)
        original = "api-key-\u00e9\u00e0\u00fc\u2603"
        encrypted = encrypt_key(original, key_path=kp)
        decrypted = decrypt_key(encrypted, key_path=kp)
        assert decrypted == original


class TestDifferentPlaintexts:
    """Different plaintexts must produce different ciphertexts."""

    def test_different_plaintexts_produce_different_ciphertexts(
        self, tmp_path: Path
    ) -> None:
        """Two different API keys should not encrypt to the same value."""
        kp = _tmp_key_path(tmp_path)
        ct1 = encrypt_key("key-alpha", key_path=kp)
        ct2 = encrypt_key("key-beta", key_path=kp)
        assert ct1 != ct2


class TestDecryptWithWrongKey:
    """Decrypting with a different key should raise ValueError."""

    def test_wrong_key_raises(self, tmp_path: Path) -> None:
        """decrypt_key raises ValueError when a different encryption key is used."""
        kp1 = tmp_path / "key1.key"
        kp2 = tmp_path / "key2.key"
        encrypted = encrypt_key("my-secret", key_path=kp1)
        with pytest.raises(ValueError, match="invalid or tampered"):
            decrypt_key(encrypted, key_path=kp2)


class TestDecryptTamperedCiphertext:
    """Tampered ciphertext should raise ValueError."""

    def test_tampered_ciphertext_raises(self, tmp_path: Path) -> None:
        """Flipping a character in the ciphertext causes a ValueError."""
        kp = _tmp_key_path(tmp_path)
        encrypted = encrypt_key("my-secret", key_path=kp)
        # Flip the last real character (before any padding '=')
        chars = list(encrypted)
        idx = len(chars) - 1
        while idx >= 0 and chars[idx] == "=":
            idx -= 1
        chars[idx] = "A" if chars[idx] != "A" else "B"
        tampered = "".join(chars)
        with pytest.raises(ValueError, match="invalid or tampered"):
            decrypt_key(tampered, key_path=kp)

    def test_garbage_ciphertext_raises(self, tmp_path: Path) -> None:
        """Completely garbage input raises ValueError."""
        kp = _tmp_key_path(tmp_path)
        # Ensure key file exists first
        encrypt_key("seed", key_path=kp)
        with pytest.raises(ValueError, match="invalid or tampered"):
            decrypt_key("not-a-valid-token-at-all!!!", key_path=kp)


class TestKeyFileLifecycle:
    """The encryption key file is created on first use and reused thereafter."""

    def test_key_file_created_on_first_use(self, tmp_path: Path) -> None:
        """Encryption key file is created automatically when it does not exist."""
        kp = _tmp_key_path(tmp_path)
        assert not kp.exists()
        encrypt_key("trigger-creation", key_path=kp)
        assert kp.exists()

    def test_key_file_reused_on_subsequent_calls(self, tmp_path: Path) -> None:
        """The same key file is reused on subsequent calls, so decrypt works."""
        kp = _tmp_key_path(tmp_path)
        ct1 = encrypt_key("hello", key_path=kp)
        key_bytes_1 = kp.read_bytes()

        # Second call should reuse the same key
        ct2 = encrypt_key("world", key_path=kp)
        key_bytes_2 = kp.read_bytes()
        assert key_bytes_1 == key_bytes_2

        # Both should decrypt correctly
        assert decrypt_key(ct1, key_path=kp) == "hello"
        assert decrypt_key(ct2, key_path=kp) == "world"


class TestBase64Safety:
    """Encrypted output must be base64-safe for SQLite TEXT storage."""

    def test_encrypted_output_is_base64_safe(self, tmp_path: Path) -> None:
        """The encrypted string should only contain URL-safe base64 characters."""
        kp = _tmp_key_path(tmp_path)
        encrypted = encrypt_key("sk-test-key-12345", key_path=kp)
        # Fernet tokens are URL-safe base64 — this should not raise
        base64.urlsafe_b64decode(encrypted)
        # Also confirm it is pure ASCII with no whitespace
        assert encrypted == encrypted.strip()
        assert encrypted.isascii()
