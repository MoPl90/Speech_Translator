import whisper
import numpy as np
from typing import Any, Dict, Optional


class WhisperModel:
    def __init__(
        self,
        name: str = "small",
        language: str = "",
        device: str = "cpu",
    ):
        self.language = language if language != "" else None
        self.model = whisper.load_model(
            f"{name}.{language}" if language != "" else f"{name}",
            download_root=f"/models/whisper_{name}_{language}"
            if language != ""
            else f"/models/whisper_{name}",
        ).to(device)
        self.device = device

    def predict(
        self, data: np.ndarray, task: str = "transcribe", language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe and optionally translate audio data using the pre-loaded Whisper model
           Supports translation to English only.

        Args:
            data (np.ndarray): Input audio data.
        """
        data = np.array(data, dtype=np.int16).flatten()
        audio = data / 32768.0 if data.max() > 1 else data

        return self.model.transcribe(
            audio.astype(np.float32),
            language=language,
            task=task,
            fp16=True if self.device == "cuda" else False,
        )
