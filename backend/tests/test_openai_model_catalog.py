"""Regression coverage for the OpenAI-compatible model catalog."""

from api.routers import openai_compat


def test_model_catalog_entries_declare_audio_capability():
    """Canvas must classify every VoiceStudio catalog entry as audio."""
    for model_kind in ("tts", "stt"):
        entry = openai_compat._model_entry("example", model_kind, True)
        assert entry["model_type"] == "audio"
