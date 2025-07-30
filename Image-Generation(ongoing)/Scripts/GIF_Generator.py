from PIL import Image
import glob
import re

def get_epoch_num(path):
    match = re.search(r'ep_(\d+)', path)
    return int(match.group(1)) if match else -1
# Path to your autoencoder outputs
image_folder = r"..\Checkpoints\CVaE\MSE_512_200\Photos"
output_folder= r"..\git_repos\DS_Test\VAE\Results\VAE_512_200_MSE.gif"
image_files = sorted(glob.glob(f"{image_folder}\\*.png"))  # or .jpg
sorted_paths = sorted(image_files, key=get_epoch_num)
# Load all frames
frames = [Image.open(img) for img in sorted_paths]

# Save as GIF
frames[0].save(
    output_folder,
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=100,   # duration per frame in ms
    loop=0          # 0 = loop forever
)

print("GIF saved as autoencoder_faces.gif")