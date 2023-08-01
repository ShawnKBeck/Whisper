from moviepy.editor import *
import streamlit as st # Import Streamlit library
import openai # Import OpenAI API library
import nltk # Import Natural Language Toolkit
import os

openai.api_key = st.secrets["OPENAI_API_KEY"]

def transcribe_audio(file_path): # Defining a function to transcribe audio using OpenAI API
  with open(file_path, "rb") as file: # Opening the audio file in binary mode for transcription
    transcription = openai.Audio.transcribe("whisper-1", file) # Using OpenAI Audio API to transcribe audio file
    return transcription["text"] # Returning the transcribed text

def summarize_text(text):  # Function to summarize text using OpenAI's GPT-3.5-turbo
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Please summarize the following text and provide bullet points: {text}"},
    ]

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    return response['choices'][0]['message']['content']

# Streamlit code

st.title('Audio Transcription App')

uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav'])

if uploaded_file is not None:
  file_extension = os.path.splitext(uploaded_file.name)[1]  # Get the file extension
  temp_file_path = f"tempfile{file_extension}"  # Create a temporary file name with the same extension

  with open(temp_file_path, "wb") as f: 
    f.write(uploaded_file.getbuffer()) # Writes the uploaded file to a temporary file
  
  transcription_text = transcribe_audio(temp_file_path) # Transcribing the audio file

  # Summarize the transcribed text and break it down into bullet points
  summary = summarize_text(transcription_text)

  nltk.download('punkt') # Downloading pre-trained sentence tokenizer

  sentences = nltk.tokenize.sent_tokenize(transcription_text) # Tokenizing transcribed text into sentences

  with open('formatted_transcription.txt', 'w') as f: # Writing sentences to a file
    for sentence in sentences:
        f.write(sentence + '\\n')

  st.write("Transcription:", transcription_text) # Printing transcription
  st.write("Summary:", summary)  # Printing the summary

def convert_mp4_to_mp3(mp4_file):
    mp3_path = mp4_file.replace('.mp4', '.mp3')
    video = VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(mp3_path)
    return mp3_path

uploaded_file = st.file_uploader("Upload an MP4 file", type="mp4")
if uploaded_file is not None:
    mp4_file_path = "uploaded.mp4"
    with open(mp4_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    mp3_file_path = convert_mp4_to_mp3(mp4_file_path)
    st.write(f"MP3 file created: {mp3_file_path}")
