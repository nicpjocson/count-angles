from collections import defaultdict

# Define face angles
face_angles = ["front", "30", "45", "60", "side", "mixed"]
counts = defaultdict(int)

# Input file
input_file = "lrs3_sorted_pretraing.txt"

# Read the file and count occurrences
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        angle = line.split("/")[0]  # Extract the face angle
        if angle in face_angles:
            counts[angle] += 1

# Print results
for angle in face_angles:
    print(f"{angle}: {counts[angle]}")