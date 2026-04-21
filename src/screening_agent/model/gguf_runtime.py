"""Local GGUF backend scaffold for future clinical analysis integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from screening_agent.graph.state import PatientRecord
from screening_agent.model.clinical_backend import ClinicalAnalysisPayload, build_clinical_input
from screening_agent.prompts import CLINICAL_ANALYSIS_SYSTEM_PROMPT


class GGUFClinicalBackend:
    """Run clinical analysis against a local GGUF model when llama-cpp is available."""

    backend_name = "gguf"

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the GGUF backend.

        Args:
            model_path: Path to the local GGUF model artifact.
        """

        self._model_path = Path(model_path)
        self._llm = self._try_create_runtime()

    def analyze(
        self,
        *,
        user_message: str,
        active_patient: PatientRecord | None,
    ) -> ClinicalAnalysisPayload:
        """Analyze a clinical request with a local GGUF runtime.

        Args:
            user_message: Latest user complaint or question.
            active_patient: Optional active patient record.

        Returns:
            Structured clinical analysis payload.

        Raises:
            RuntimeError: If llama-cpp is unavailable or the result is not valid JSON.
        """

        if self._llm is None:
            raise RuntimeError(
                "GGUF runtime is unavailable because llama-cpp-python is not installed.",
            )
        completion = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": CLINICAL_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_clinical_input(user_message, active_patient),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = completion["choices"][0]["message"]["content"]
        return ClinicalAnalysisPayload.model_validate(json.loads(content))

    def _try_create_runtime(self) -> Any | None:
        """Create the optional llama-cpp runtime when available.

        Returns:
            A llama-cpp model instance or `None`.
        """

        try:
            from llama_cpp import Llama
        except ImportError:
            return None
        return Llama(
            model_path=str(self._model_path),
            n_ctx=4096,
            chat_format="chatml",
            verbose=False,
        )
