

import glob
import os
import sys
sys.path.append('C:/Users/aiane/git_repos/DS_Test/audio_project/Scripts')
from definitions import split_audio_by_silence_and_duration, infer_text


def process_audio(input_path) :
    output_folder = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Data_splits'
    transcript_folder = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Transcriptions'

    for file in glob.glob(os.path.join(output_folder, "*.wav")):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    for filename in os.listdir(transcript_folder):
            file_path1 = os.path.join(transcript_folder, filename)
            os.remove(file_path1)
    # Clear the transcription file
   
    
    
    filename = os.path.splitext(os.path.basename(input_path))[0]

    split_audio_by_silence_and_duration(input_path, output_folder)



    folder_path = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Data_splits'
    file_paths = glob.glob(os.path.join(folder_path, "*.wav"))

# Print all file paths
    output_file = r'C:\Users\aiane\git_repos\DS_Test\audio_project\Transcriptions\transcriptions.txt'

    with open(output_file, "a", encoding="utf-8") as f:
        for path in file_paths:
            try:
                result = infer_text(path)
                filename = os.path.basename(path)
                f.write(f"{result}")
                print(f"Done: {filename}")
            except Exception as e:
                print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_audio(sys.argv[1])
    else:
        print("Please provide the input audio file path")                