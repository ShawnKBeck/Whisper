from pydub import AudioSegment
import streamlit as st # Import Streamlit library
import openai # Import OpenAI API library
import nltk # Import Natural Language Toolkit
import os

openai.api_key = st.secrets["OPENAI_API_KEY"]

def split_audio(file_path): # Function to split audio file into chunks
    file_extension = os.path.splitext(file_path)[1]  # Get the file extension
    if file_extension == '.mp3':
        song = AudioSegment.from_mp3(file_path)
    elif file_extension == '.wav':
        song = AudioSegment.from_wav(file_path)
    elif file_extension == '.m4a':
        song = AudioSegment.from_file(file_path, 'm4a')
    ten_minutes = 10 * 60 * 1000
    first_10_minutes = song[:ten_minutes]
    first_10_minutes.export("tempfile.mp3", format="mp3")
    return "tempfile.mp3"

def transcribe_audio(file_path): # Defining a function to transcribe audio using OpenAI API
    split_file_path = split_audio(file_path)
    with open(split_file_path, "rb") as file: # Opening the audio file in binary mode for transcription
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

uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'm4a'])

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


