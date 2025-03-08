import os
import zipfile

# Define the folder containing the zip files
zip_folder = "C:/Users/nicpj/Desktop/New folder/AY 24-25/thesis/datasets/OuluVS2/OuluVS2/OuluVS2-zip"  # Change this to your actual path

# List of indicated speakers
# indicated_speakers = {1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 53}
# indicated_speakers = {6, 8, 9, 15, 26, 30, 34, 43, 44, 49, 51, 52}
indicated_speakers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53}


# Counters and unique value sets
total_videos = 0
unique_speakers = set()
unique_views = set()
unique_utterances = set()

# Loop through all zip files in the folder
for zip_filename in os.listdir(zip_folder):
    if zip_filename.startswith("orig_s") and zip_filename.endswith(".zip"):
        speaker_id = zip_filename.split("_s")[-1].split(".zip")[0]

        # Check if speaker ID is in the indicated list
        if speaker_id.isdigit() and int(speaker_id) in indicated_speakers:
            zip_path = os.path.join(zip_folder, zip_filename)

            # Open the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_name in zip_file.namelist():
                    # Extract the actual filename, ignoring the outer folder
                    base_name = os.path.basename(file_name)  # Get only the filename
                    parts = base_name.split("_")

                    # Ensure the filename matches "s<id>_v<angle>_u<utterance>.mp4"
                    if len(parts) >= 3 and parts[0].startswith("s") and parts[1].startswith("v") and parts[2].startswith("u") and base_name.endswith(".mp4"):
                        total_videos += 1

                        # Extract speaker, view, and utterance IDs
                        spk_id = parts[0][1:]  # Remove 's' from speaker ID
                        view_id = parts[1][1:]  # Remove 'v' from view ID
                        utt_id = parts[2][1:].split(".")[0]  # Remove 'u' from utterance ID and drop '.mp4'

                        unique_speakers.add(spk_id)
                        unique_views.add(view_id)
                        unique_utterances.add(utt_id)

# Print the total number of videos
print(f"Total number of videos: {total_videos}")

# Print unique values
print(f"Unique speakers: {', '.join(sorted(unique_speakers, key=int))}")
print(f"Unique views: {', '.join(sorted(unique_views, key=int))}")
print(f"Unique utterances: {', '.join(sorted(unique_utterances, key=int))}")
