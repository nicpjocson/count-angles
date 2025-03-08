import os
import shutil

def move_non_pretrain_videos(outer_folder):
    pretrain_folder = os.path.join(outer_folder, "sorted_videos", "pretrainDataList")
    train_folder = os.path.join(outer_folder, "sorted_videos", "trainDataList")
    
    if not os.path.exists(pretrain_folder):
        print(f"Pretrain folder not found: {pretrain_folder}")
        return
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)
    
    for angle in ["front", "30", "45", "60", "side", "mixed"]:
        pretrain_angle_folder = os.path.join(pretrain_folder, angle)
        train_angle_folder = os.path.join(train_folder, angle)
        
        if not os.path.exists(pretrain_angle_folder):
            continue
        
        os.makedirs(train_angle_folder, exist_ok=True)
        
        for file in os.listdir(pretrain_angle_folder):
            if not file.startswith("pretrain_"):
                src = os.path.join(pretrain_angle_folder, file)
                dst = os.path.join(train_angle_folder, file)
                shutil.move(src, dst)
                print(f"Moved {file} to {dst}")
    
    print("Non-pretrain files moved to train folder!")

# Example usage
outer_folder = "C:/Users/nicpj/Desktop/New folder/AY 24-25/thesis/datasets/lrs2_classified"
move_non_pretrain_videos(outer_folder)
