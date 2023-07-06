import streamlit as st # Import Streamlit library
import openai # Import OpenAI API library
import nltk # Import Natural Language Toolkit
import os


my_secret = os.environ['OPENAI_API_KEY']
openai.api_key = my_secret # Authenticating with OpenAI API key

def transcribe_audio(file_path): # Defining a function to transcribe audio using OpenAI API
  with open(file_path, "rb") as file: # Opening the audio file in binary mode for transcription
    transcription = openai.Audio.transcribe("whisper-1", file) # Using OpenAI Audio API to transcribe audio file
    return transcription["text"] # Returning the transcribed text

# Streamlit code

st.title('Audio Transcription App')

uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav'])

if uploaded_file is not None:
  with open("tempfile", "wb") as f: 
    f.write(uploaded_file.getbuffer()) # Writes the uploaded file to a temporary file
  transcription_text = transcribe_audio("tempfile") # Transcribing the audio file

  nltk.download('punkt') # Downloading pre-trained sentence tokenizer

  sentences = nltk.tokenize.sent_tokenize(transcription_text) # Tokenizing transcribed text into sentences

  with open('formatted_transcription.txt', 'w') as f: # Writing sentences to a file
    for sentence in sentences:
        f.write(sentence + '\n')

  st.write("Transcription:", transcription_text) # Printi