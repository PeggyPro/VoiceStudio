"""Catalogue Torch Whisper picks must persist and load the selected local model."""
from __future__ import annotations

import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_REPO = "openai/whisper-large-v3"
_ENV = "OMNIVOICE_PYTORCH_ASR_MODEL"
_CT2 = "Systran/faster-whisper-small"


@pytest.fixture
def selection(monkeypatch, tmp_path):
    from api.routers import engines
    from core import prefs
    from services import asr_backend as ab

    monkeypatch.setattr(prefs, "_PREFS_PATH", str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "_EXTERNALLY_PROVIDED", frozenset())
    for key in (_ENV, "OMNIVOICE_ASR_BACKEND", "ASR_MODEL_FASTER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    # Setter writes are also undone by monkeypatch's environment restoration.
    monkeypatch.setenv(_ENV, "")
    monkeypatch.setenv("ASR_MODEL_FASTER", "")
    monkeypatch.setattr(ab, "_capture_backend", None)
    monkeypatch.setattr(ab, "_capture_backend_key", None)
    rows = [{"id": name, "display_name": name, "available": True,
             "routing_status": "cpu_only", "effective_device": "cpu"}
            for name in ("pytorch-whisper", "faster-whisper", "faster-whisper-isolated",
                         "parakeet-mlx", "moonshine")]
    monkeypatch.setattr(ab, "list_backends", lambda: rows)

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("Torch model selection must not access the network")

    connect = socket.socket.connect

    def local_socketpair_only(sock, address):
        # Windows asyncio implements socketpair with a loopback connection.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return connect(sock, address)
        return refuse_network()

    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket.socket, "connect", local_socketpair_only)
    app = FastAPI()
    app.include_router(engines.router)
    client = TestClient(app, client=("127.0.0.1", 12345))
    return SimpleNamespace(client=client, prefs=prefs, ab=ab, engines=engines)


def _snapshot(cache, repo=_REPO, *, sharded=False):
    snapshot = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "local-revision"
    snapshot.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] if sharded else ["model.safetensors"]
    for name in names:
        with (snapshot / name).open("wb") as handle:
            handle.truncate(6 * 1024 * 1024)
    if sharded:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": names[0], "b": names[1]}}), encoding="utf-8",
        )
    return snapshot


def _select(selection, repo=_REPO, backend="pytorch-whisper"):
    return selection.client.post("/engines/select", json={
        "family": "asr", "backend_id": backend, "model_id": repo,
    })


def test_native_selection_persists_applies_and_survives_env_restore(selection, tmp_path):
    _snapshot(tmp_path / "empty-hub")
    response = _select(selection)
    assert response.status_code == 200, response.text
    assert response.json()["active"] == "pytorch-whisper"
    saved = selection.prefs._load()
    assert saved["env." + _ENV] == _REPO
    assert saved["asr_backend"] == "pytorch-whisper"
    assert os.environ[_ENV] == _REPO
    assert selection.ab.PyTorchWhisperBackend._model_name() == _REPO
    assert selection.ab._offline_asr_repo("pytorch-whisper") == _REPO
    assert selection.client.get("/engines/asr").json()["active_model"] == _REPO

    # Simulate startup in a new process with the persisted file and no env pin.
    os.environ.pop(_ENV)
    selection.prefs.restore_env(saved)
    assert os.environ[_ENV] == _REPO
    assert not selection.prefs.is_env_shadowed(_ENV)
    assert selection.client.get("/engines/asr").json()["active_model"] == _REPO


@pytest.mark.parametrize("damage", ["absent", "config-only", "weights-only", "missing-tokenizer",
                                   "tiny-weight", "missing-shard", "tiny-shard", "bad-index"])
def test_native_selection_rejects_incomplete_installs_without_writing(selection, tmp_path, damage):
    if damage != "absent":
        snapshot = _snapshot(tmp_path / "empty-hub", sharded="shard" in damage or damage == "bad-index")
        if damage == "config-only":
            (snapshot / "model.safetensors").unlink()
        elif damage == "weights-only":
            for name in ("config.json", "preprocessor_config.json", "tokenizer.json"):
                (snapshot / name).unlink()
        elif damage == "missing-tokenizer":
            (snapshot / "tokenizer.json").unlink()
        elif damage == "tiny-weight":
            (snapshot / "model.safetensors").write_bytes(b"partial")
        elif damage == "missing-shard":
            (snapshot / "model-00002-of-00002.safetensors").unlink()
        elif damage == "tiny-shard":
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"partial")
        elif damage == "bad-index":
            (snapshot / "model.safetensors.index.json").write_text("{", encoding="utf-8")
    response = _select(selection)
    assert response.status_code == 409, response.text
    assert selection.prefs._load() == {}
    assert not os.environ[_ENV]


@pytest.mark.parametrize("repo,backend", [(_CT2, "pytorch-whisper"),
    ("mlx-community/whisper-large-v3-turbo", "pytorch-whisper"),
    ("someone/unknown-whisper", "pytorch-whisper"),
    ("k2-fsa/OmniVoice", "pytorch-whisper"),
    (_REPO, "faster-whisper"), (_REPO, "parakeet-mlx"), (_REPO, "moonshine")])
def test_selection_rejects_unknown_or_incompatible_models(selection, repo, backend):
    response = _select(selection, repo, backend)
    assert response.status_code == 400, response.text
    assert selection.prefs._load() == {}


def test_external_model_pin_remains_authoritative_and_does_not_persist(selection, tmp_path, monkeypatch):
    _snapshot(tmp_path / "empty-hub")
    monkeypatch.setenv(_ENV, "openai/whisper-small")
    monkeypatch.setattr(selection.prefs, "_EXTERNALLY_PROVIDED", frozenset({_ENV}))
    response = _select(selection)
    assert response.status_code == 409, response.text
    assert _ENV in response.json()["detail"]
    assert selection.prefs._load() == {}
    assert os.environ[_ENV] == "openai/whisper-small"
    assert selection.ab._offline_asr_repo("pytorch-whisper") == "openai/whisper-small"


@pytest.mark.parametrize("backend", ["faster-whisper", "faster-whisper-isolated"])
def test_existing_ct2_selection_is_preserved(selection, tmp_path, monkeypatch, backend):
    from api.routers.setup import models

    monkeypatch.setattr(models, "is_cached", lambda repo: repo == _CT2)
    response = _select(selection, _CT2, backend)
    assert response.status_code == 200, response.text
    assert selection.prefs.get("asr_model_faster") == _CT2
    assert os.environ["ASR_MODEL_FASTER"] == _CT2
    assert selection.prefs.get("env." + _ENV) is None


def test_saved_model_resolves_before_startup_restore_and_loads_local_snapshot(selection, tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "empty-hub")
    selection.prefs.set_("env." + _ENV, _REPO)
    loaded = []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **kw: loaded.append(kw) or object()))
    monkeypatch.setattr(selection.ab.PyTorchWhisperBackend, "_pick_device", lambda *a: "cpu")
    selection.ab.PyTorchWhisperBackend().ensure_loaded()
    assert loaded[0]["model"] == str(snapshot)
    assert selection.ab._offline_asr_repo("pytorch-whisper") == _REPO


def test_new_selection_releases_cached_native_and_tts_asr_pipelines(selection, tmp_path, monkeypatch):
    from services import model_manager

    _snapshot(tmp_path / "empty-hub")
    old = selection.ab.PyTorchWhisperBackend(asr_pipe=object())
    monkeypatch.setattr(selection.ab, "_capture_backend", old)
    monkeypatch.setattr(selection.ab, "_capture_backend_key", "auto")
    tts = SimpleNamespace(_asr_pipe=object())
    monkeypatch.setattr(model_manager, "model", tts)
    response = _select(selection)
    assert response.status_code == 200, response.text
    assert selection.ab._capture_backend is None
    assert tts._asr_pipe is None


def test_complete_sharded_native_model_can_be_selected(selection, tmp_path):
    _snapshot(tmp_path / "empty-hub", sharded=True)
    response = _select(selection)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("preferred", ["model.safetensors", "model.safetensors.index.json"])
def test_incomplete_preferred_format_cannot_hide_behind_complete_bin(selection, tmp_path, preferred):
    snapshot = _snapshot(tmp_path / "empty-hub")
    (snapshot / "pytorch_model.bin").write_bytes((snapshot / "model.safetensors").read_bytes())
    (snapshot / "model.safetensors").unlink()
    (snapshot / preferred).write_bytes(b"partial")
    response = _select(selection)
    assert response.status_code == 409, response.text
    assert selection.prefs._load() == {}


def test_legacy_bin_weights_and_vocab_tokenizer_remain_supported(selection, tmp_path):
    snapshot = _snapshot(tmp_path / "empty-hub")
    (snapshot / "pytorch_model.bin").write_bytes((snapshot / "model.safetensors").read_bytes())
    (snapshot / "model.safetensors").unlink()
    (snapshot / "tokenizer.json").unlink()
    (snapshot / "vocab.json").write_text("{}", encoding="utf-8")
    (snapshot / "merges.txt").write_text("#version: 0.2", encoding="utf-8")
    response = _select(selection)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("configured", ["someone/custom-whisper", "/local/custom-whisper"])
def test_custom_external_override_still_wins_over_saved_pick(selection, monkeypatch, configured):
    selection.prefs.set_("env." + _ENV, _REPO)
    monkeypatch.setenv(_ENV, configured)
    assert selection.ab.PyTorchWhisperBackend._model_name() == configured
    assert selection.ab._offline_asr_repo("pytorch-whisper") == configured


@pytest.mark.parametrize("engine_id", ["omnivoice", "omnivoice-subprocess"])
def test_running_omnivoice_children_are_retired_without_interrupting_active_jobs(selection, tmp_path, monkeypatch, engine_id):
    from engines.omnivoice_subprocess import OmniVoiceMPSSubprocessBackend, OmniVoiceSubprocessBackend
    from services import tts_backend as tts

    _snapshot(tmp_path / "empty-hub")
    unloaded = []

    base = OmniVoiceMPSSubprocessBackend if engine_id == "omnivoice" else OmniVoiceSubprocessBackend

    class FakeChild(base):
        def __init__(self):
            pass

        def unload(self):
            unloaded.append(self)

    active, cached = FakeChild(), FakeChild()
    monkeypatch.setattr(tts, "_active_instance", active)
    monkeypatch.setattr(tts, "_active_instance_id", engine_id)
    monkeypatch.setattr(tts, "_ENGINE_INSTANCES", {FakeChild: cached})
    monkeypatch.setattr(tts, "_ENGINE_IN_USE", {})
    monkeypatch.setattr(tts, "_RETIRED_ENGINES", {})
    monkeypatch.setattr(tts, "_ENGINE_LAST_USED", {})
    with tts.engine_in_use(cached):
        response = _select(selection)
        assert response.status_code == 200, response.text
        assert tts._active_instance is None
        assert FakeChild not in tts._ENGINE_INSTANCES
        assert unloaded == []
    assert active in unloaded and cached in unloaded


def test_native_selection_preserves_other_tts_instances(selection, tmp_path, monkeypatch):
    from services import tts_backend as tts

    _snapshot(tmp_path / "empty-hub")

    class OtherEngine:
        id = "kittentts"

        def unload(self):
            pytest.fail("An ASR model pick must preserve unrelated TTS models")

    other = OtherEngine()
    monkeypatch.setattr(tts, "_active_instance", other)
    monkeypatch.setattr(tts, "_active_instance_id", "kittentts")
    monkeypatch.setattr(tts, "_ENGINE_INSTANCES", {OtherEngine: other})
    response = _select(selection)
    assert response.status_code == 200, response.text
    assert tts._active_instance is other
    assert tts._ENGINE_INSTANCES[OtherEngine] is other


def test_omnivoice_child_preload_uses_saved_local_pick(selection, tmp_path, monkeypatch):
    from engines.omnivoice_subprocess import main as child
    from services import model_manager
    from utils import hf_progress

    snapshot = _snapshot(tmp_path / "empty-hub")
    selection.prefs.set_("env." + _ENV, _REPO)
    preloads, options = [], []
    model = SimpleNamespace(load_asr_model=lambda model_name=None: preloads.append(model_name))

    class FakeOmniVoice:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            options.append(kwargs)
            if kwargs.get("load_asr"):
                model.load_asr_model()
            return model

    monkeypatch.setattr(child, "_model", None)
    monkeypatch.setattr(child, "_prefetch_weights", lambda *_args: None)
    monkeypatch.setattr(child, "_send", lambda *_args: None)
    monkeypatch.setattr(model_manager, "_lazy_torch", lambda: SimpleNamespace(float16="float16"))
    monkeypatch.setattr(model_manager, "_lazy_omnivoice", lambda: FakeOmniVoice)
    monkeypatch.setattr(model_manager, "resolve_omnivoice_checkpoint", lambda: "test")
    monkeypatch.setattr(model_manager, "get_best_device", lambda: "mps")
    monkeypatch.setattr(model_manager, "should_preload_tts_asr", lambda: True)
    monkeypatch.setattr(hf_progress, "register_listener", lambda _listener: 1)
    monkeypatch.setattr(hf_progress, "unregister_listener", lambda _listener: None)
    child._load_model(object())
    assert options[0]["load_asr"] is False
    assert preloads == [str(snapshot)]


def test_selection_checks_model_acceptance_before_persisting(selection, tmp_path, monkeypatch):
    from services import model_acceptance

    _snapshot(tmp_path / "empty-hub")
    checked = []

    def require_acceptance(repos):
        checked.extend(repos)
        raise ValueError("Model licence acceptance required")

    monkeypatch.setattr(model_acceptance, "ensure_accepted", require_acceptance)
    response = _select(selection)
    assert response.status_code == 409, response.text
    assert checked == [_REPO]
    assert selection.prefs._load() == {}
    assert not os.environ[_ENV]
