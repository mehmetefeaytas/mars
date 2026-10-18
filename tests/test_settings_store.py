# SPDX-License-Identifier: AGPL-3.0-only
"""Calisma-zamani ayar deposu ve ayar uclari testleri."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mars import api
from mars.config import load_settings
from mars.settings_store import (
    load_overrides,
    mask_secret,
    public_settings,
    save_overrides,
    store_path,
)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_DATA_DIR", str(tmp_path))
    return tmp_path / "settings.local.json"


def test_save_and_load_overrides(tmp_store):
    save_overrides({"llm_model": "qwen2.5:3b-instruct", "llm_timeout": 42,
                    "llm_enabled": True}, tmp_store)
    data = load_overrides(tmp_store)
    assert data["llm_model"] == "qwen2.5:3b-instruct"
    assert data["llm_timeout"] == 42
    assert data["llm_enabled"] is True


def test_unknown_fields_ignored(tmp_store):
    save_overrides({"not_a_field": 1, "llm_model": "x"}, tmp_store)
    data = load_overrides(tmp_store)
    assert "not_a_field" not in data
    assert data["llm_model"] == "x"


def test_empty_secret_keeps_existing(tmp_store):
    save_overrides({"llm_api_key": "evren_llm_secret"}, tmp_store)
    save_overrides({"llm_api_key": ""}, tmp_store)
    assert load_overrides(tmp_store)["llm_api_key"] == "evren_llm_secret"


def test_mask_secret():
    assert mask_secret("") == ""
    assert mask_secret("evren_llm_abcdef123") == "••••••f123"
    assert mask_secret("short") == "••••••"


def test_public_settings_masks_key(tmp_store):
    save_overrides({"llm_api_key": "evren_llm_abcdef123"}, tmp_store)
    pub = public_settings(load_overrides(tmp_store))
    assert pub["llm_api_key"].startswith("••••••")
    assert "abcdef" not in pub["llm_api_key"]


def test_load_settings_applies_overrides(tmp_store):
    save_overrides({"llm_model": "model-x", "llm_provider": "evren"}, tmp_store)
    s = load_settings()
    assert s.llm_model == "model-x"
    assert s.llm_provider == "evren"


def test_store_path_uses_data_dir(tmp_store):
    assert store_path() == tmp_store


# --- API uclari ---
def test_settings_endpoints(tmp_store):
    client = TestClient(api.app)
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "current" in body and "providers" in body
    assert {p["id"] for p in body["providers"]} >= {"ollama", "evren", "openai", "custom"}

    patch = {"llm_provider": "evren", "llm_model": "qwen3.8-flash-next",
             "llm_api_key": "evren_llm_testkey123"}
    r2 = client.put("/api/settings", json=patch)
    assert r2.status_code == 200
    assert r2.json()["current"]["llm_provider"] == "evren"
    assert r2.json()["current"]["llm_api_key"].startswith("••••••")

    # Provider degisince varsayilan base URL uygulanir
    assert "evren-llmapi" in r2.json()["current"]["llm_base_url"]

    saved = json.loads(tmp_store.read_text(encoding="utf-8"))
    assert saved["llm_model"] == "qwen3.8-flash-next"
    assert saved["llm_api_key"] == "evren_llm_testkey123"
