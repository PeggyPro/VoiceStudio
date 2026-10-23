"""The engine inventory identifies configured TTS models before first synthesis."""

import pytest


@pytest.mark.parametrize(
    "engine,model_env,override,default,expected_override",
    [
        (
            "cosyvoice", "OMNIVOICE_COSYVOICE_MODEL",
            "custom/CosyVoice2-0.5B", "Fun-CosyVoice3-0.5B", "CosyVoice2-0.5B",
        ),
        (
            "voxcpm2", "OMNIVOICE_VOXCPM_MODEL",
            "custom/VoxCPM", "openbmb/VoxCPM2", "custom/VoxCPM",
        ),
        (
            "kittentts", "OMNIVOICE_KITTENTTS_MODEL",
            "custom/kitten-tts-nano", "KittenML/kitten-tts-mini-0.8",
            "custom/kitten-tts-nano",
        ),
        (
            "moss-tts-nano", "OMNIVOICE_MOSS_TTS_MODEL",
            "custom/MOSS", "OpenMOSS-Team/MOSS-TTS-Nano", "custom/MOSS",
        ),
    ],
)
@pytest.mark.parametrize("configured", [False, True], ids=["default", "env-override"])
def test_tts_active_model_before_first_load(
    engine, model_env, override, default, expected_override, configured, monkeypatch,
):
    from api.routers import engines
    from services import tts_backend
    from services.subprocess_backend import SubprocessBackend

    if configured:
        monkeypatch.setenv(model_env, override)
    else:
        monkeypatch.delenv(model_env, raising=False)
    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: engine)
    # Availability and language metadata are covered separately. Identity must
    # not depend on either installed packages or an already-created instance.
    monkeypatch.setattr(tts_backend, "list_backends", lambda: [])
    monkeypatch.setattr(tts_backend, "_active_instance", None)
    monkeypatch.setattr(tts_backend, "_active_instance_id", None)

    def unexpected_construction(*args, **kwargs):
        pytest.fail("Reading configured identity must not construct a TTS worker")

    monkeypatch.setattr(SubprocessBackend, "__init__", unexpected_construction)
    for cls in (
        tts_backend.CosyVoiceBackend, tts_backend.VoxCPM2Backend,
        tts_backend.KittenTTSBackend, tts_backend.MossTTSNanoBackend,
    ):
        monkeypatch.setattr(cls, "__init__", unexpected_construction)
        monkeypatch.setattr(cls, "_ensure_loaded", unexpected_construction)

    payload = engines._family_payload("tts", tts_backend)

    assert payload["active"] == engine
    assert payload["active_model"] == (expected_override if configured else default)
    assert tts_backend._active_instance is None
