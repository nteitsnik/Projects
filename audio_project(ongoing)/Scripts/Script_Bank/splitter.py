from pydub import AudioSegment
from pydub.silence import detect_silence
import os

def split_audio_by_silence_and_duration(audio_path, output_dir, min_duration=25_000, max_duration=30_000, min_silence_len=700, silence_thresh_offset=16):
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(audio_path, format="m4a")
    silence_thresh = audio.dBFS - silence_thresh_offset
    total_length = len(audio)
    
    start = 0
    chunk_index = 0

    while start < total_length:
        # Define max end time for chunk
        end = min(start + max_duration, total_length)

        # Look for silences between [min_duration, max_duration]
        silence_ranges = detect_silence(
            audio[start:end],
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        # Find first suitable silence >= min_duration
        split_point = None
        for silence_start, silence_end in silence_ranges:
            if silence_start >= min_duration:
                split_point = start + silence_start
                break

        if split_point is None:
            # No silence found → hard cut at max_duration
            split_point = end

        chunk = audio[start:split_point]
        out_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}.wav")
        chunk.export(out_path, format="wav")
        print(f"Saved chunk {chunk_index}: {out_path} [{len(chunk)/1000:.2f}s]")
        
        chunk_index += 1
        start = split_point

# --- Usage ---


input_path = r'C:\Users\Nteit\audio_project\Data\Recording.m4a'
filename = os.path.splitext(os.path.basename(input_path))[0]
print(filename)
output_folder = fr'C:\Users\Nteit\audio_project\Data_splits'
split_audio_by_silence_and_duration(input_path, output_folder)