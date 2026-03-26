"""Tests for settings CRUD and provider/preferences helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from opinionforge.models.config import ProviderConfig, UserPreferences
from opinionforge.storage.database import Database
from opinionforge.storage.settings import SettingsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide a connected and initialised temp Database."""
    db_path = tmp_path / "settings_test.db"
    database = Database(db_path)
    await database.connect()
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
async def store(db: Database) -> SettingsStore:
    """Provide a SettingsStore backed by the temp Database."""
    return SettingsStore(db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_set_and_get(store: SettingsStore) -> None:
    """A stored setting can be retrieved by key."""
    await store.set("theme", "dark")
    assert await store.get("theme") == "dark"


async def test_get_nonexistent_returns_none(store: SettingsStore) -> None:
    """Getting a key that does not exist returns None."""
    assert await store.get("nonexistent_key") is None


async def test_upsert_overwrites(store: SettingsStore) -> None:
    """Setting the same key twice overwrites the previous value."""
    await store.set("theme", "light")
    await store.set("theme", "dark")
    assert await store.get("theme") == "dark"


async def test_get_all(store: SettingsStore) -> None:
    """get_all returns all stored settings as a dict."""
    await store.set("a", "1")
    await store.set("b", "2")
    await store.set("c", "3")
    all_settings = await store.get_all()
    assert all_settings == {"a": "1", "b": "2", "c": "3"}


async def test_get_all_empty(store: SettingsStore) -> None:
    """get_all returns an empty dict when no settings exist."""
    assert await store.get_all() == {}


async def test_delete(store: SettingsStore) -> None:
    """delete() removes a setting and returns True."""
    await store.set("tmp", "value")
    assert await store.delete("tmp") is True
    assert await store.get("tmp") is None


async def test_delete_nonexistent_returns_false(store: SettingsStore) -> None:
    """delete() returns False when the key does not exist."""
    assert await store.delete("no_such_key") is False


async def test_get_provider_config_round_trip(store: SettingsStore) -> None:
    """ProviderConfig can be saved and retrieved as a Pydantic model."""
    config = ProviderConfig(
        provider_type="anthropic",
        model="claude-3-opus-20240229",
        api_key="sk-test-12345",
        base_url=None,
        max_tokens=8192,
    )
    await store.set_provider_config(config)
    loaded = await store.get_provider_config()
    assert loaded is not None
    assert loaded.provider_type == "anthropic"
    assert loaded.model == "claude-3-opus-20240229"
    assert loaded.api_key == "sk-test-12345"
    assert loaded.max_tokens == 8192


async def test_set_provider_config_stores_json(store: SettingsStore) -> None:
    """set_provider_config stores the config as a JSON string."""
    config = ProviderConfig(
        provider_type="ollama",
        model="llama3",
    )
    await store.set_provider_config(config)
    raw = await store.get("provider_config")
    assert raw is not None
    assert '"provider_type": "ollama"' in raw or '"provider_type":"ollama"' in raw


async def test_get_provider_config_none_when_empty(store: SettingsStore) -> None:
    """get_provider_config returns None when nothing has been stored."""
    assert await store.get_provider_config() is None


async def test_get_user_preferences_returns_defaults(store: SettingsStore) -> None:
    """get_user_preferences returns defaults when nothing is stored."""
    prefs = await store.get_user_preferences()
    assert prefs.default_mode == "analytical"
    assert prefs.default_stance == 0
    assert prefs.default_intensity == 0.5
    assert prefs.default_length == "standard"
    assert prefs.theme == "light"
    assert prefs.auto_launch is False
    assert prefs.onboarding_completed is False


async def test_set_user_preferences_round_trip(store: SettingsStore) -> None:
    """UserPreferences can be saved and retrieved as a Pydantic model."""
    prefs = UserPreferences(
        default_mode="polemical",
        default_stance=-30,
        default_intensity=0.8,
        default_length="long",
        theme="dark",
        auto_launch=True,
        onboarding_completed=True,
    )
    await store.set_user_preferences(prefs)
    loaded = await store.get_user_preferences()
    assert loaded.default_mode == "polemical"
    assert loaded.default_stance == -30
    assert loaded.default_intensity == 0.8
    assert loaded.default_length == "long"
    assert loaded.theme == "dark"
    assert loaded.auto_launch is True
    assert loaded.onboarding_completed is True
