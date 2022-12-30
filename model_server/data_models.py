from pydantic import BaseModel
from typing import List


class AudioInputChunk(BaseModel):
    """Features for audio chunk"""

    data: List[int] | List[List[int]]
    sample_rate: int = 16000


class TextInputChunk(BaseModel):
    """Features for audio chunk"""

    text: str | List[str]
    sample_rate: int = 22050


class TranslationChunk(BaseModel):
    """Features for audio chunk"""

    text: str | List[str]
    language: str 


class Task(BaseModel):
    """Celery task representation"""

    task_id: str
    status: str


class WhisperPrediction(BaseModel):
    """Prediction task result"""

    task_id: str
    status: str
    text: str

class TTSPrediction(BaseModel):
    """Prediction task result"""

    task_id: str
    status: str
    audio: List[int] | List[List[int]]
    
class GPTPrediction(BaseModel):
    """Prediction task result"""

    task_id: str
    status: str
    text: str
