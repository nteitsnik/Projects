import librosa
import subprocess
import math
import wave

result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
print(result.stdout)


def get_duration_ffprobe(filename):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filename
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def split_audio_to_wav_with_overlap(input_file, output_pattern, chunk_length=30, overlap=3):
    duration = get_duration_ffprobe(input_file)
    step = chunk_length - overlap
    num_chunks = math.ceil((duration - overlap) / step)

    for i in range(num_chunks):
        start = i * step
        output_file = output_pattern % i
        
        cmd = [
            'ffmpeg',
            '-y',                # overwrite output files without asking
            '-nostdin',          # prevent ffmpeg from waiting for input
            '-ss', str(start),   # start time
            '-t', str(chunk_length), # duration of chunk
            '-i', input_file,    # input file
            '-acodec', 'pcm_s16le',  # WAV codec
            '-ar', '44100',          # sample rate
            output_file
        ]
        print(f"Creating chunk {i}: start={start}s, file={output_file}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print(f"Error splitting chunk {i}: {proc.stderr.decode()}")


input_path = r'C:\Users\Nteit\audio_project\Data\Recording.m4a'
filename = os.path.splitext(os.path.basename(input_path))[0]
print(filename)
output_folder = fr'C:\Users\Nteit\audio_project\Data_splits\{filename}_chunk_%03d.wav'
split_audio_to_wav_with_overlap(input_path, output_folder)
