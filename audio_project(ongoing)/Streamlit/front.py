import os
import streamlit as st
from main import process_audio  # Import your function

UPLOAD_DIR = r'C:\Users\Nteit\audio_project\Data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("🎵 Audio Upload App")

audio_file = st.file_uploader("Upload your audio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    file_path = os.path.join(UPLOAD_DIR, audio_file.name)
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(file_path)
    st.success(f"Saved to {file_path}")

    # Call the processing function
    process_audio(file_path)
    st.success("Audio processing completed!")