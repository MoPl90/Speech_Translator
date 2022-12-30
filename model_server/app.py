from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from typing import Optional

from ml_worker_app.tasks import whisper_inference, text_to_speech, gpt_prompt
from data_models import (
    AudioInputChunk,
    Task,
    WhisperPrediction,
    TextInputChunk,
    TTSPrediction,
    TranslationChunk,
    GPTPrediction,
)

app = FastAPI()

#################################
#
#
#       Whisper
#
#
#################################


@app.post("/transcribe/predict", response_model=Task, status_code=202)
async def transcribe(
    audio: AudioInputChunk, task: str = "transcribe", language: Optional[str] = None
):
    """Create celery prediction task. Return task_id to client in order to retrieve result"""
    task_id = whisper_inference.delay(audio.data, task=task, language=language)
    return {"task_id": str(task_id), "status": "Processing"}


@app.get(
    "/transcribe/result/{task_id}/",
    response_model=WhisperPrediction,
    status_code=200,
    responses={202: {"model": Task, "description": "Accepted: Not Ready"}},
)
async def whisper_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():

        return JSONResponse(
            status_code=202, content={"task_id": str(task_id), "status": "Processing"}
        )
    result = task.get()

    return JSONResponse(
        status_code=200, content={"task_id": task_id, "status": "Success", **result}
    )


#################################
#
#
#       Translation
#
#
#################################


@app.post("/translation/predict", response_model=Task, status_code=202)
async def translate(text: TranslationChunk, target_language: str = "en"):
    """Create celery prediction task. Return task_id to client in order to retrieve result"""
    task_id = gpt_prompt.delay(
        text.text, source_language=text.language, target_language=target_language
    )
    return {"task_id": str(task_id), "status": "Processing"}


@app.get(
    "/translation/result/{task_id}/",
    response_model=GPTPrediction,
    status_code=200,
    responses={202: {"model": Task, "description": "Accepted: Not Ready"}},
)
async def openai_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():

        return JSONResponse(
            status_code=202, content={"task_id": str(task_id), "status": "Processing"}
        )
    result = task.get()
    print(result)
    
    return JSONResponse(
        status_code=200, content={"task_id": task_id, "status": "Success", **result}
    )


#################################
#
#
#       Vocalization
#
#
#################################


@app.post("/tts/predict", response_model=Task, status_code=202)
async def vocalize(text: TextInputChunk):
    """Create celery prediction task. Return task_id to client in order to retrieve result"""
    task_id = text_to_speech.delay(text.text)
    return {"task_id": str(task_id), "status": "Processing"}


@app.get(
    "/tts/result/{task_id}/",
    response_model=TTSPrediction,
    status_code=200,
    responses={202: {"model": Task, "description": "Accepted: Not Ready"}},
)
async def tts_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():

        return JSONResponse(
            status_code=202, content={"task_id": str(task_id), "status": "Processing"}
        )
    result = task.get()

    return JSONResponse(
        status_code=200, content={"task_id": task_id, "status": "Success", **result}
    )
