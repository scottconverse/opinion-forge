"""API key encryption/decryption using Fernet symmetric encryption.

Provides encrypt_key() and decrypt_key() functions for securing API keys
at rest in the SQLite database.  The encryption key is derived from a
machine-specific secret file stored in the application data directory.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from platformdirs import user_data_dir

# Default location for the encryption key file.
_DEFAULT_KEY_DIR = Path(user_data_dir("opinionforge"))
_DEFAULT_KEY_FILE = _DEFAULT_KEY_DIR / "secret.key"


def _get_or_create_key(key_path: Path | None = None) -> bytes:
    """Load the Fernet key from disk, creating it if it does not exist.

    The key file is a 44-byte URL-safe base64-encoded Fernet key.  If
    the file does not exist it is generated automatically on first use.

    Args:
        key_path: Optional override for the key file location.
            Defaults to ``~/.opinionforge/secret.key`` (platform-dependent).

    Returns:
        The raw Fernet key bytes.
    """
    path = key_path or _DEFAULT_KEY_FILE
    if path.exists():
        return path.read_bytes().strip()

    # Generate a new Fernet key and persist it.
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    path.write_bytes(key)
    return key


def _get_fernet(key_path: Path | None = None) -> Fernet:
    """Return a Fernet instance backed by the stored encryption key.

    Args:
        key_path: Optional override for the key file location.

    Returns:
        A :class:`cryptography.fernet.Fernet` instance.
    """
    return Fernet(_get_or_create_key(key_path))


def encrypt_key(plaintext: str, *, key_path: Path | None = None) -> str:
    """Encrypt an API key string using Fernet symmetric encryption.

    The returned ciphertext is a URL-safe base64-encoded string that is
    safe to store in SQLite text columns.

    Args:
        plaintext: The API key to encrypt.
        key_path: Optional override for the encryption key file location.

    Returns:
        A Fernet-encrypted, base64-safe string.
    """
    f = _get_fernet(key_path)
    token = f.encrypt(plaintext.encode("utf-8"))
    return token.decode("ascii")


def decrypt_key(ciphertext: str, *, key_path: Path | None = None) -> str:
    """Decrypt a Fernet-encrypted API key string.

    Args:
        ciphertext: The encrypted string produced by :func:`encrypt_key`.
        key_path: Optional override for the encryption key file location.

    Returns:
        The original plaintext API key.

    Raises:
        ValueError: If the ciphertext is invalid or has been tampered with.
    """
    f = _get_fernet(key_path)
    try:
        plaintext_bytes = f.decrypt(ciphertext.encode("ascii"))
    except (InvalidToken, Exception) as exc:
        raise ValueError(
            "Failed to decrypt API key — ciphertext is invalid or tampered."
        ) from exc
    return plaintext_bytes.decode("utf-8")
