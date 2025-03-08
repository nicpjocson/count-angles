import os
import shutil
import re

def sort_videos(outer_folder):
    source_folder = os.path.join(outer_folder, "videos")
    file_lists_folder = os.path.join(outer_folder, "file_lists")
    destination_folder = os.path.join(outer_folder, "sorted_videos")
    
    # Define sets
    sets = ["pretrainDataList", "trainDataList", "valDataList", "testDataList"]
    
    # Create destination folders
    for set_name in sets:
        set_path = os.path.join(destination_folder, set_name)
        os.makedirs(set_path, exist_ok=True)
        for angle in ["front", "30", "45", "60", "side", "mixed"]:
            os.makedirs(os.path.join(set_path, angle), exist_ok=True)
    
    # Read file lists and move files
    for set_name in sets:
        list_path = os.path.join(file_lists_folder, f"{set_name}.txt")
        if not os.path.exists(list_path):
            print(f"File list {list_path} not found. Skipping...")
            continue
    
        with open(list_path, "r") as file:
            filenames = set()
            for line in file:
                base_name = line.strip().split()[0]  # Take only the first part before space
                filenames.add(base_name + ".mp4")  # Standard format
                filenames.add(f"pretrain_{base_name}.mp4")  # Accommodate pretrain-specific format
    
        for angle in ["front", "30", "45", "60", "side", "mixed"]:
            angle_folder = os.path.join(source_folder, angle)
            if not os.path.exists(angle_folder):
                continue
    
            for file in os.listdir(angle_folder):
                if file in filenames:
                    src = os.path.join(angle_folder, file)
                    dst = os.path.join(destination_folder, set_name, angle, file)
                    shutil.move(src, dst)
                    print(f"Moved {file} to {dst}")
    
    print("Sorting complete!")

# Example usage
outer_folder = "C:/Users/nicpj/Desktop/New folder/AY 24-25/thesis/datasets/lrs2_classified"
sort_videos(outer_folder)
