<h1 align=center > Speech-to-speech translation agent </h1>

<h2> Description </h2>

This project contains a simple, yet self-contained speech-to-speech translation agent. It uses OpenAI's Whisper model for transcription and translation to Egnlish, OpenAI's GPT-3 for translation to the target language, and different TTS engines for the final speech output.

The first version uses the following TTS engines:
 - Tacotron2 for spectrogram generation
 - Vocoder/HifiGAN for audio generation

Both models are trained on the LJSpeech dataset and used off the shelf from the `speechbrain` library.



<h2> Set-up</h2>

To get started, you need to clone the repository and install the requirements from the `environment.yml` file. The easiest way to do this is to use conda: `conda env create -f environment.yml`.

Next, you need to set up the `.env` file. This file contains the API keys for the different services used in the project. You can get the API keys from the following services:

- [OpenAI](https://openai.com/)


The backend runs in a dockerized environment. It is wrapped in an API running a celery task queue. To start the model server backend, you need to run the following command (`docker-compose` needs to be installed):

``` 
> docker-compose build
> docker-compose up -d
```


The backend is now running on port 5000. To test it, you can use the notebook `4.0-mp-test_celery_app.ipynb` in the `notebooks` folder.


Mac OS:

To get the audio working on Mac OS, you need to install pulseaudio and start it as a service. This is because the default audio output on Mac OS is not compatible with the `sounddevice` library.

- Install pulseaudio [`> brew install pulseaudio`]
- start pulseaudio [`> brew services start pulseaudio`]


<h2> To-Dos </h2>

- Add more TTS engines
- CDUA support
- ONNX models
- training of vocoder and TTS models on more data