"""Read-only capture metadata follows auto routing, not saved recorder prefs."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import dictation as dr
from services import asr_backend as ab


@pytest.fixture
def capture_client(monkeypatch):
    store = {
        dr.PREF_ENABLED: True,
        dr.PREF_MODEL_ID: "sherpa-whisper-tiny",
    }
    monkeypatch.setattr(dr.prefs, "get", lambda key, default=None: store.get(key, default))
    monkeypatch.delenv("OMNIVOICE_SHERPA_ASR_MODEL", raising=False)
    monkeypatch.delenv("ASR_MODEL_PARAKEET_MLX", raising=False)
    monkeypatch.setattr(ab, "_capture_backend", None)
    monkeypatch.setattr(ab, "_capture_backend_key", None)
    monkeypatch.setattr(ab, "_capture_prefers_parakeet", lambda: False)
    monkeypatch.setattr(ab.MLXWhisperBackend, "is_available", lambda: (False, "unavailable"))
    monkeypatch.setattr(ab.FasterWhisperBackend, "is_available", lambda: (True, "ready"))
    monkeypatch.setattr(ab.PyTorchWhisperBackend, "is_available", lambda: (True, "ready"))
    monkeypatch.setattr(ab, "faster_whisper_model_id", lambda: "Systran/faster-whisper-small")
    monkeypatch.setattr(ab, "pytorch_whisper_model_id", lambda: "openai/whisper-small")
    monkeypatch.setattr(ab, "_repo_installed", lambda _repo: True)
    monkeypatch.setattr(dr.sd, "sherpa_available", lambda: (True, "ready"))
    monkeypatch.setattr(dr.sd, "is_installed", lambda _spec: True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Listing capture metadata must not construct/load/download or mutate state")

    for cls in (
        ab.SherpaDictationBackend, ab.ParakeetMLXBackend, ab.MLXWhisperBackend,
        ab.FasterWhisperBackend, ab.PyTorchWhisperBackend,
    ):
        monkeypatch.setattr(cls, "__init__", forbidden)
        monkeypatch.setattr(cls, "ensure_loaded", forbidden)
    monkeypatch.setattr(ab, "get_capture_asr_backend", forbidden)
    monkeypatch.setattr(ab, "_touch_capture", forbidden)
    monkeypatch.setattr(dr.prefs, "set_", forbidden)
    monkeypatch.setattr(dr.sd, "build_online_recognizer", forbidden)
    monkeypatch.setattr(dr.sd, "build_offline_recognizer", forbidden)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)

    app = FastAPI()
    app.include_router(dr.router)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client, store


def _selection(client):
    response = client.get("/dictation/models")
    assert response.status_code == 200
    return response.json()["capture_selection"]


@pytest.mark.parametrize("installed", [True, False])
def test_saved_sherpa_selection_is_canonical_and_reports_install_state(capture_client, monkeypatch, installed):
    client, store = capture_client
    spec = dr.sd.get_spec("sherpa-zipformer-en-20m")
    store[dr.PREF_MODEL_ID] = spec.repo_id
    monkeypatch.setattr(dr.sd, "is_installed", lambda _spec: installed)

    assert _selection(client) == {
        "model_id": spec.id, "repo_id": spec.repo_id, "label": spec.label,
        "available": installed,
    }
    assert client.get("/dictation/prefs").json()["model_id"] == spec.repo_id


def test_env_override_wins_over_disabled_demoted_saved_selection(capture_client, monkeypatch):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    spec = dr.sd.get_spec("sherpa-zipformer-en-20m")
    store[dr.sd.PREF_SILENT_MODELS] = [store[dr.PREF_MODEL_ID], spec.id]
    monkeypatch.setenv("OMNIVOICE_SHERPA_ASR_MODEL", spec.repo_id)

    assert _selection(client) == {
        "model_id": spec.id, "repo_id": spec.repo_id, "label": spec.label,
        "available": True,
    }
    assert client.get("/dictation/prefs").json()["model_id"] == "sherpa-whisper-tiny"


@pytest.mark.parametrize("reason", ["disabled", "demoted", "unavailable", "invalid_env"])
@pytest.mark.parametrize("installed", [True, False])
def test_sherpa_bypass_reports_capture_fallback(capture_client, monkeypatch, reason, installed):
    client, store = capture_client
    if reason == "disabled":
        store[dr.PREF_ENABLED] = False
    elif reason == "demoted":
        store[dr.sd.PREF_SILENT_MODELS] = [store[dr.PREF_MODEL_ID]]
    elif reason == "unavailable":
        monkeypatch.setattr(dr.sd, "sherpa_available", lambda: (False, "not installed"))
    else:
        monkeypatch.setenv("OMNIVOICE_SHERPA_ASR_MODEL", "unknown-model")
    monkeypatch.setattr(ab, "_repo_installed", lambda _repo: installed)

    assert _selection(client) == {
        "model_id": None, "repo_id": "Systran/faster-whisper-small", "available": installed,
    }
    assert client.get("/dictation/prefs").json()["model_id"] == "sherpa-whisper-tiny"


@pytest.mark.parametrize("backend", ["parakeet", "mlx", "pytorch"])
def test_fallback_chain_uses_its_actual_model(capture_client, monkeypatch, backend):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    if backend == "parakeet":
        monkeypatch.setattr(ab, "_capture_prefers_parakeet", lambda: True)
        monkeypatch.setenv("ASR_MODEL_PARAKEET_MLX", "local-org/parakeet-custom")
        repo = "local-org/parakeet-custom"
    elif backend == "mlx":
        monkeypatch.setattr(ab.MLXWhisperBackend, "is_available", lambda: (True, "ready"))
        repo = ab._MLX_MODEL_TURBO
    else:
        monkeypatch.setattr(ab.FasterWhisperBackend, "is_available", lambda: (False, "unavailable"))
        repo = "openai/whisper-small"

    assert _selection(client) == {"model_id": None, "repo_id": repo, "available": True}


def test_unavailable_pytorch_fallback_is_unknown_without_changing_preflight(capture_client, monkeypatch):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    monkeypatch.setattr(ab.FasterWhisperBackend, "is_available", lambda: (False, "unavailable"))
    monkeypatch.setattr(ab.PyTorchWhisperBackend, "is_available", lambda: (False, "unavailable"))

    assert _selection(client) == {"model_id": None, "repo_id": None, "available": False}
    # Existing missing-model preflight keeps its historical last-resort repo.
    assert ab._capture_whisper_repo() == "openai/whisper-small"


def test_unknown_faster_whisper_alias_is_not_replaced_with_default(capture_client, monkeypatch):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    monkeypatch.setattr(ab, "faster_whisper_model_id", lambda: "unmapped-alias")

    assert _selection(client) == {"model_id": None, "repo_id": None, "available": False}


def test_probe_failure_is_unknown_and_does_not_leak_diagnostics(capture_client, monkeypatch):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False

    def broken_probe():
        raise RuntimeError("private local path or token")

    monkeypatch.setattr(ab, "_capture_prefers_parakeet", broken_probe)
    assert _selection(client) == {"model_id": None, "repo_id": None, "available": False}


def test_warm_fallback_reports_model_the_capture_singleton_will_reuse(capture_client, monkeypatch):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    backend = object.__new__(ab.FasterWhisperBackend)
    backend._model_name = "Systran/faster-whisper-tiny"
    monkeypatch.setattr(ab, "_capture_backend", backend)
    monkeypatch.setattr(ab, "_capture_backend_key", "auto:parakeet=0")

    assert _selection(client) == {
        "model_id": None, "repo_id": "Systran/faster-whisper-tiny", "available": True,
    }
    assert ab._capture_backend is backend
    # Existing cold preflight resolution remains unchanged.
    assert ab._capture_whisper_repo() == "Systran/faster-whisper-small"


@pytest.mark.parametrize("loaded_repo", ["openai/whisper-tiny", None])
def test_warm_pytorch_pipeline_does_not_invent_current_pref_model(capture_client, monkeypatch, loaded_repo):
    client, store = capture_client
    store[dr.PREF_ENABLED] = False
    backend = object.__new__(ab.PyTorchWhisperBackend)
    backend._pipe = SimpleNamespace(model=SimpleNamespace(name_or_path=loaded_repo))
    monkeypatch.setattr(ab, "_capture_backend", backend)
    monkeypatch.setattr(ab, "_capture_backend_key", "auto:parakeet=0")

    assert _selection(client) == {
        "model_id": None, "repo_id": loaded_repo, "available": loaded_repo is not None,
    }
