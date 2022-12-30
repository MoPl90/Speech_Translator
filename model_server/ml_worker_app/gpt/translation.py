import os
import openai
import numpy as np
from typing import Any, Dict, Optional

openai.api_key = os.environ["OPENAI_API_KEY"]


class GPTModel:
    def __init__(
        self,
        model_name: str = "text-davinci-002",  # Cheapo model, otherwise use text-davinci-002
    ):
        self.model_name = model_name

    def predict(
        self, text: str, source_language: str, target_language: str = "en"
    ) -> Dict[str, Any]:

        prompt = f"Translate from {source_language} to {target_language}: '{text}'"

        print(openai.api_key)
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=1.0,
        )

        return response["choices"][0]
