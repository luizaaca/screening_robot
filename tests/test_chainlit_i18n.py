"""Tests for Chainlit bilingual UI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app_chainlit


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (None, "pt-BR"),
        ("", "pt-BR"),
        ("pt", "pt-BR"),
        ("pt-BR", "pt-BR"),
        ("pt-br", "pt-BR"),
        ("pt_BR", "pt-BR"),
        ("en", "en-US"),
        ("en-US", "en-US"),
        ("en-GB", "en-US"),
        ("en-US,en;q=0.9", "en-US"),
        ("fr-FR", "pt-BR"),
    ],
)
def test_normalize_locale(language: str | None, expected: str) -> None:
    """Browser language codes should collapse to the supported UI locales."""

    assert app_chainlit._normalize_locale(language) == expected


def test_build_welcome_message_defaults_to_pt_br() -> None:
    """The app-controlled welcome message should fall back to pt-BR."""

    message = app_chainlit._build_welcome_message(3)

    assert "# Assistente de Triagem Clínica" in message
    assert "Registros de pacientes disponíveis no momento: 3." in message
    assert "micção frequente" in message


def test_build_welcome_message_supports_en_us() -> None:
    """English browsers should receive English app-controlled welcome text."""

    message = app_chainlit._build_welcome_message(0, locale="en-US")

    assert "# Clinical Screening Assistant" in message
    assert "Current patient records available: 0." in message
    assert "The database is currently empty." in message


def test_app_controlled_upload_and_progress_text_are_localized() -> None:
    """Upload prompts and progress labels should follow the normalized locale."""

    assert app_chainlit._ui_string("pt-BR", "upload_prompt").startswith(
        "Envie um arquivo de vídeo"
    )
    assert app_chainlit._ui_string("en-US", "upload_prompt").startswith(
        "Upload one video file"
    )
    assert app_chainlit._ui_string("unknown", "upload_timeout").startswith(
        "O envio do vídeo"
    )
    assert app_chainlit._progress_node_label("video_analysis", "pt-BR") == (
        "Processando vídeo"
    )
    assert app_chainlit._progress_node_label("video_analysis", "en-US") == (
        "Processing video"
    )
    assert app_chainlit._progress_error_output("Falha 12345678", "pt-BR") == (
        "Erro técnico: Falha ****5678"
    )


def test_pt_br_translation_matches_en_us_structure() -> None:
    """The pt-BR Chainlit translation must keep the en-US key structure."""

    en_us = json.loads(
        (REPO_ROOT / ".chainlit" / "translations" / "en-US.json").read_text(
            encoding="utf-8"
        )
    )
    pt_br = json.loads(
        (REPO_ROOT / ".chainlit" / "translations" / "pt-BR.json").read_text(
            encoding="utf-8"
        )
    )

    assert _json_shape(pt_br) == _json_shape(en_us)


def test_chainlit_markdown_locale_files_exist() -> None:
    """Chainlit should find both explicit locale readme files."""

    markdown_file_names = {path.name for path in REPO_ROOT.glob("chainlit*.md")}

    assert (REPO_ROOT / "chainlit.md").is_file()
    assert (REPO_ROOT / "chainlit_pt-BR.md").is_file()
    assert (REPO_ROOT / "chainlit_en-US.md").is_file()
    assert "chainlit_pt-BR.md" in markdown_file_names
    assert "chainlit_pt-br.md" not in markdown_file_names


def _json_shape(value: object) -> object:
    """Return a JSON-compatible structural shape for translation comparisons."""

    if isinstance(value, dict):
        return {key: _json_shape(nested_value) for key, nested_value in value.items()}
    return type(value).__name__
