import pandas as pd
import librosa
import glob
from pydub import AudioSegment
from pydub.silence import detect_silence
import os
import torchaudio
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from definitions import split_audio_by_silence_and_duration, infer_text

folder_path = r"C:\Users\Nteit\audio_project\Data_splits"
file_paths = glob.glob(os.path.join(folder_path, "*.wav"))


output_file = r"C:\Users\Nteit\audio_project\Transcriptions\transcriptions.txt"

with open(output_file, "a", encoding="utf-8") as f:
        for path in file_paths:
            try:
                result = infer_text(path)
                filename = os.path.basename(path)
                f.write(f"{filename}:\n{result}\n\n")
                print(f"Done: {filename}")
            except Exception as e:
                print(f"Failed on {path}: {e}")
