# Voice_assistance_using_LLM

import os
import time
import wikipedia
import requests
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import whisper

# Constants
DURATION = 5
SAMPLE_RATE = 16000
RECORDINGS_DIR = Path("recordings")
TRANSCRIPTS_DIR = Path("transcripts")
RECORDINGS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# Load models
print("🔄 Loading Whisper model...")
asr_model = whisper.load_model("base")

print("🔄 Loading QA model...")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Weather API
OPENWEATHER_API_KEY = "3d9b094a1892b4f3bdba6c7413526260"
DEFAULT_CITY = "Delhi"


def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print("🎧 Recording... Please speak.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RECORDINGS_DIR / f"audio_{timestamp}.wav"
    wav.write(filename, fs, audio)
    print(f"✅ Audio saved: {filename}")
    return filename


def transcribe_audio(audio_path):
    print("📜 Transcribing audio...")
    result = asr_model.transcribe(str(audio_path))
    transcript = result["text"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = TRANSCRIPTS_DIR / f"transcript_{timestamp}.txt"
    transcript_path.write_text(transcript)
    print(f"📄 Transcript saved: {transcript_path}")
    print(f"🗣️ You said: {transcript}")
    return transcript


def speak_text(text):
    print("🔊 Speaking...")
    tts = gTTS(text=text, lang='en')
    temp_path = "temp_audio.mp3"
    tts.save(temp_path)
    sound = AudioSegment.from_mp3(temp_path)
    play(sound)
    os.remove(temp_path)
    print("✅ Done speaking.")


def get_weather(city=DEFAULT_CITY):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return "Sorry, I couldn't fetch the weather data."
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The current temperature in {city} is {temp}°C with {desc}."
    except Exception:
        return "Weather information is currently unavailable."


def get_wikipedia_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your question is ambiguous. Try being more specific: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "I couldn't find any relevant information on Wikipedia."
    except Exception:
        return "Something went wrong while fetching Wikipedia data."


def generate_context(prompt):
    prompt_lower = prompt.lower()
    if "weather" in prompt_lower:
        return get_weather()
    elif any(x in prompt_lower for x in ["who", "what", "when", "where", "tell me about"]):
        return get_wikipedia_summary(prompt)
    elif "president" in prompt_lower:
        return "The President of India is the ceremonial head of state. The current president is Droupadi Murmu."
    elif "prime minister" in prompt_lower:
        return "The Prime Minister of India is Narendra Modi."
    elif "capital" in prompt_lower:
        return "The capital of India is New Delhi."
    else:
        return "I'm not sure, but I can try looking it up."


def generate_response(prompt):
    print("🤖 Generating response...")
    try:
        context = generate_context(prompt)
        result = qa_pipeline(question=prompt, context=context)
        answer = result["answer"]
        print(f"🤖 Assistant: {answer}")
        return answer
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        return "I'm sorry, I couldn't find an answer."


if __name__ == "__main__":
    audio_path = record_audio()
    if audio_path:
        prompt = transcribe_audio(audio_path)
        if prompt:
            answer = generate_response(prompt)
            speak_text(answer)
