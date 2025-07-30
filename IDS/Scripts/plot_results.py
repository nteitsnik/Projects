import os
import json
import pandas as pd
import random 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Path to your folder containing JSON files
folder_path = r"C:\Users\aiane\git_repos\DS_Test\IDS\Results"

# Find all JSON files in the folder
json_files = glob.glob(os.path.join(folder_path, "*.json"))

# List to hold rows
data_rows = []

for file_path in json_files:
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Load JSON
    with open(file_path, "r") as f:
        metrics = json.load(f)

    # Append metrics as a row
    data_rows.append({
        "Model": filename,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1 Score": metrics["f1_score"]
    })

# Create DataFrame
df = pd.DataFrame(data_rows)

# Optional: set Model as index
df.set_index("Model", inplace=True)

# Show DataFrame
print(df)





df = df.sort_values("Accuracy", ascending=False)
df1=df.copy()
df1['Accuracy']=df1['Accuracy']*100
# Set a clean style
sns.set(style="whitegrid")

# Create a color palette
palette = sns.color_palette("viridis", len(df))

# Create the figure
plt.figure(figsize=(10, 6))

# Horizontal barplot
bars = sns.barplot(
    x="Accuracy",
    y="Model",
    data=df1,
    palette=palette
)

# Add data labels on bars
for i, (value, name) in enumerate(zip(df["Test Accuracy"], df["Model"])):
    plt.text(
        value + 0.005,  # Slightly offset to the right
        i,
        f"{value:.2%}",  # Show as percentage
        va="center",
        fontsize=12,
        fontweight="bold",
        color="black"
    )
plt.xlim(85,97)
# Titles and labels
plt.title("Model Test Accuracies", fontsize=18, fontweight="bold")
plt.xlabel("Accuracy (%)", fontsize=14)
plt.ylabel("Model", fontsize=14)

# Remove spines for a cleaner look
sns.despine(left=True, bottom=True)

# Tight layout
plt.tight_layout()

# Show the plot
full_path = os.path.join(folder_path, 'output_filename.png')
plt.savefig(full_path, dpi=300, bbox_inches="tight")

output_txt_path = os.path.join(folder_path, "output_dataframe.txt")
df.to_csv(output_txt_path, sep=",", index=True)



