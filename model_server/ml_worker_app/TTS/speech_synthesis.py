from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import numpy as np
from typing import Any, Dict, Optional


class TTSModel:
    def __init__(
        self,
        tts_model: str = "speechbrain/tts-tacotron2-ljspeech",
        vocoder_model: str = "speechbrain/tts-hifigan-ljspeech",
        device: str = "cpu",
    ):

        # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
        self.tts_model = Tacotron2.from_hparams(
            source=tts_model, savedir="/models/tts"
        ).to(device=device)
        self.vocoder = HIFIGAN.from_hparams(
            source=vocoder_model, savedir="/models/vocoder"
        ).to(device=device)
        self.device = device

    def predict(self, text: str | list[str]) -> Dict[str, Any]:
        """Synthesize speech from text"""

        # Running the TTS
        mel_output, mel_length, alignment = self.tts_model.encode_text(text)

        # Running Vocoder (spectrogram-to-waveform)
        waveforms = self.vocoder.decode_batch(mel_output)

        return {
            "waveforms": waveforms.detach().cpu().numpy().tolist(),
            "alignment": alignment.detach().cpu().numpy().tolist(),
            "mel_length": mel_length.detach().cpu().numpy().tolist(),
        }
