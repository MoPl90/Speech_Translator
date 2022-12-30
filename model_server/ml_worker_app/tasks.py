import importlib
import logging
from celery import Task
import os
from typing import List, Any, Dict

from .worker import app


class ModelInferenceTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """

    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            logging.info("Loading Model...")
            module_import = importlib.import_module(self.path[0])
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj(**self.model_args)
            logging.info("Model loaded")
        return self.run(*args, **kwargs)


@app.task(
    ignore_result=False,
    bind=True,
    base=ModelInferenceTask,
    path=("ml_worker_app.transcribe", "WhisperModel"),
    model_args={
        "name": "base",
        "language": "",
        "device": "cpu",
    },
    name="{}.{}".format(__name__, "Whisper"),
)
def whisper_inference(
    self, data: List[int], task="transcribe", language=None
) -> Dict[str, Any]:
    """
    Transcribe audio data using the Whisper model
    """

    return self.model.predict([data], task=task, language=language) | {"task": task}


@app.task(
    ignore_result=False,
    bind=True,
    base=ModelInferenceTask,
    path=("ml_worker_app.TTS", "TTSModel"),
    model_args={
        "tts_model": "speechbrain/tts-tacotron2-ljspeech",
        "vocoder_model": "speechbrain/tts-hifigan-ljspeech",
        "device": "cpu",
    },
    name="{}.{}".format(__name__, "TTS"),
)
def text_to_speech(self, text: str | List[str]) -> Dict[str, Any]:
    """
    Synthesize speech from text using the TTS model(s)
    """
    return self.model.predict(text)


@app.task(
    ignore_result=False,
    bind=True,
    base=ModelInferenceTask,
    path=("ml_worker_app.gpt", "GPTModel"),
    model_args={
    #     "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "model_name": "text-curie-001",
    },
    name="{}.{}".format(__name__, "GPT"),
)
def gpt_prompt(
    self, text: str, source_language: str, target_language: str
) -> Dict[str, Any]:
    """
    Synthesize speech from text using the TTS model(s)
    """
    return self.model.predict(text, source_language, target_language)
