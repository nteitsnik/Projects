
import torchaudio
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load audio using librosa (or torchaudio)
def infer_text(path) :
    speech, sampling_rate = librosa.load(path, sr=16000)



    speech_tensor = torch.tensor(speech)

# Must be 2D input: (batch, time)
    speech_tensor = speech_tensor.unsqueeze(0).to(device)

# Prepare input with attention mask
    inputs = processor(
        speech.tolist(),  # pass raw audio as a list of floats
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True
    )
# Send tensors to device
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

# Generate predictions
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask
        )

# Decode output
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


import glob
import os

folder_path = r"C:\Users\Nteit\audio_project\Data_splits"
file_paths = glob.glob(os.path.join(folder_path, "*.wav"))

# Print all file paths
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