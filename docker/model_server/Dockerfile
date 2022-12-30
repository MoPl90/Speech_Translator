FROM python:3.10.1-slim-buster

# set environment variables
WORKDIR /app

RUN apt-get update && apt-get install -y git

# Install dependencies
RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install git+https://github.com/openai/whisper.git 

COPY model_server/requirements.txt .
RUN pip install -r requirements.txt

# COPY model_server/ /app
VOLUME [ "/models" ]

# Run the application
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "app:app"]