import os
import streamlit as st
from main import process_audio
from Mistralv2 import full_pipeline
import time
import psutil


UPLOAD_DIR = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Data'
TRANSCRIPT_FILE = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Transcriptions\transcriptions.txt'

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("🎵 Audio to Text Dummy App")

audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    file_path = os.path.join(UPLOAD_DIR, audio_file.name)
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(file_path)
    st.success(f"Saved to {file_path}")

    # Call the processing function
    process_audio(file_path)
    st.success("✅ Audio processing completed!")

    # Show transcript if it exists
    if os.path.exists(TRANSCRIPT_FILE):
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        st.subheader("📝 Transcription Preview")
        st.text_area("Transcription", transcript_text, height=300)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
               label="📥 Download Transcription",
               data=transcript_text,
               file_name="transcription.txt",
               mime="text/plain"
               )
        with col2:
            process_clicked = st.button("Process Transcription")    
        if process_clicked :
            with st.spinner("This will take a while..."):
                translation = full_pipeline(transcript_text)
           
                st.subheader("Processed")
                st.write(translation)
                st.download_button(
                        label="📥 Download Summary",
                        data=translation,
                        file_name="Summarized.txt",
                        mime="text/plain"
                               )

with st.sidebar:
    if st.button("Exit App"):
        st.warning("The app will shut down in 2 seconds...")
        time.sleep(2)
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()
                