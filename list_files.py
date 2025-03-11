import os

# Define possible values
# sets = ["pretrainDataList", "testDataList", "trainDataList", "valDataList"]
sets = ["lrs3_test_classified", "lrs3_trainval_classified"]
facial_angles = ["front", "30", "45", "60", "side", "mixed"]

# Input folder and output file
# input_folder = "C:/Users/nicpj/Desktop/New folder/AY 24-25/thesis/datasets/lrs2_classified/sorted_videos"
input_folder = "C:/Users/nicpj/Desktop/New folder/AY 24-25/thesis/datasets/lrs3_classified"
# output_file = "lrs2_sorted.txt"
output_file = "lrs3_sorted_test_trainval.txt"

# Collect file paths
file_paths = []
for set_name in sets:
    for angle in facial_angles:
        folder_path = os.path.join(input_folder, set_name, angle)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = f"{set_name}/{angle}/{filename}"
                file_paths.append(file_path)

# Save to output file
with open(output_file, "w") as f:
    for path in file_paths:
        f.write(path + "\n")

print(f"File paths saved to {output_file}")
