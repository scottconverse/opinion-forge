"""CRUD operations for the settings table (key-value store)."""

from __future__ import annotations

import json
from typing import Any

from opinionforge.models.config import ProviderConfig, UserPreferences
from opinionforge.storage.database import Database

# Internal keys used by the typed helpers.
_PROVIDER_CONFIG_KEY = "provider_config"
_USER_PREFERENCES_KEY = "user_preferences"


class SettingsStore:
    """Key-value settings store backed by the ``settings`` table.

    In addition to raw key/value CRUD, provides typed helpers for
    :class:`ProviderConfig` and :class:`UserPreferences` that
    automatically serialise/deserialise to JSON.

    Args:
        db: An initialised and connected :class:`Database`.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- raw CRUD -------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Retrieve a setting value by key.

        Args:
            key: The setting key.

        Returns:
            The string value, or *None* if the key does not exist.
        """
        row = await self._db.execute_fetchone(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        """Insert or update a setting.

        Args:
            key: The setting key.
            value: The string value to store.
        """
        await self._db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            (key, value),
        )
        await self._db.commit()

    async def get_all(self) -> dict[str, str]:
        """Return all settings as a dictionary.

        Returns:
            A ``{key: value}`` mapping of every stored setting.
        """
        rows = await self._db.execute_fetchall(
            "SELECT key, value FROM settings ORDER BY key"
        )
        return {row[0]: row[1] for row in rows}

    async def delete(self, key: str) -> bool:
        """Delete a setting by key.

        Args:
            key: The setting key to remove.

        Returns:
            *True* if the key existed and was deleted, *False* otherwise.
        """
        cursor = await self._db.execute(
            "DELETE FROM settings WHERE key = ?", (key,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # -- typed helpers --------------------------------------------------------

    async def get_provider_config(self) -> ProviderConfig | None:
        """Deserialise and return the stored provider configuration.

        Returns:
            A :class:`ProviderConfig` instance, or *None* if no
            configuration has been saved yet.
        """
        raw = await self.get(_PROVIDER_CONFIG_KEY)
        if raw is None:
            return None
        data: dict[str, Any] = json.loads(raw)
        return ProviderConfig(**data)

    async def set_provider_config(self, config: ProviderConfig) -> None:
        """Serialise and store a provider configuration.

        Args:
            config: The :class:`ProviderConfig` to persist.
        """
        await self.set(_PROVIDER_CONFIG_KEY, json.dumps(config.model_dump()))

    async def get_user_preferences(self) -> UserPreferences:
        """Return stored user preferences, falling back to defaults.

        If no preferences have been saved yet, a :class:`UserPreferences`
        with all default values is returned.

        Returns:
            A :class:`UserPreferences` instance.
        """
        raw = await self.get(_USER_PREFERENCES_KEY)
        if raw is None:
            return UserPreferences()
        data: dict[str, Any] = json.loads(raw)
        return UserPreferences(**data)

    async def set_user_preferences(self, prefs: UserPreferences) -> None:
        """Serialise and store user preferences.

        Args:
            prefs: The :class:`UserPreferences` to persist.
        """
        await self.set(_USER_PREFERENCES_KEY, json.dumps(prefs.model_dump()))
