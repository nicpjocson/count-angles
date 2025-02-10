import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def classify_angle(y):
    # 90 DEGREES (SIDE)
    if y < -22 or y > 22:
        return "side"
    # 60 DEGREES
    elif y < -18 or y > 18:
        return "60"
    # 45 DEGREES
    elif y < -14 or y > 14:
        return "45"
    # 30 DEGREES
    elif y < -7 or y > 8:
        return "30"
    else:
        return "front"

    # # side
    # if y < -60 or y > 60:
    #     return "side"
    # # 60 degrees
    # elif -60 < y <= -45 or 45 <= y < 60:
    #     return "60"
    # # 45 degrees
    # elif -45 < y <= -30 or 30 <= y < 45:
    #     return "45"
    # # 30 degrees
    # elif -30 < y <= -15 or 15 < y < 30:
    #     return "30"
    # # front
    # elif -15 <= y <= 15:
    #     return "front"
    # else:
    #     return "unknown"

def process_videos(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for video_name in files:
            if video_name.endswith((".mp4")):
                video_path = os.path.join(root, video_name)
                classify_video(video_path, output_folder)

def classify_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    angle_counts = {
        "front": 0,
        "side": 0,
        "30": 0,
        "45": 0,
        "60": 0
    }

    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1

        # preprocess frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False

        # process with mediapipe
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                img_h, img_w, _ = image.shape
                face_3d = []
                face_2d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z * 3000])

                if face_2d and face_3d:
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # camera parameters
                    focal_length = img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                           [0, focal_length, img_h / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                        y_angle = angles[1]
                        detected_angle = classify_angle(y_angle)
                        angle_counts[detected_angle] += 1

                        # draw Y angle on the frame
                        cv2.putText(image, f"Y-Angle: {y_angle:.2f}°", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # draw X, Y, Z axes
                        # we will draw lines in the 2D image using the rotation matrix
                        axis = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=np.float64)
                        img_points, _ = cv2.projectPoints(axis, rot_vec, trans_vec, cam_matrix, dist_matrix)

                        img_points = np.int32(img_points).reshape(-1, 2)

                        # draw the 3D axes (X, Y, Z) on the image
                        cv2.line(image, tuple(img_points[0]), tuple(img_points[1]), (255, 0, 0), 5)  # X axis (blue)
                        cv2.line(image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 5)  # Y axis (green)
                        cv2.line(image, tuple(img_points[0]), tuple(img_points[3]), (0, 0, 255), 5)  # Z axis (red)

    # convert back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    # DEBUGGING
    for angle, count in angle_counts.items():
        print(f"{angle}: {count}")

    # Calculate percentage of frames for each angle
    angle_percentages = {angle: (count / frame_count) * 100 for angle, count in angle_counts.items()}

    # Check for dominant angle
    dominant_angle = max(angle_counts, key=angle_counts.get)
    dominant_percentage = angle_percentages[dominant_angle]

    # Classification logic
    if dominant_percentage > 80:
        # If one angle dominates >80% of the frames
        classification = dominant_angle
    else:
        # Identify angles that occur in more than 15% of frames
        significant_angles = [angle for angle, percentage in angle_percentages.items() if percentage > 15]

        if len(significant_angles) > 1:
            classification = "mixed"
        else:
            classification = dominant_angle

    # move video to classified folder
    output_subfolder = os.path.join(output_folder, classification)
    move_video(video_path, output_subfolder)

    print(f"Classified {os.path.basename(video_path)} as {classification}")

def move_video(video_path, output_folder):
    # get name of folder where video is located
    subfolder_name = os.path.basename(os.path.dirname(video_path))
    
    # ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # get base filename and extension of the video
    base_name = os.path.basename(video_path)
    file_name, file_extension = os.path.splitext(base_name)
    
    # format new video name as "<subfolder>_<orig_video_name>"
    # to avoid overwriting
    new_file_name = f"{subfolder_name}_{file_name}{file_extension}"
    
    # check if a file with the same name already exists in output folder
    new_file_path = os.path.join(output_folder, new_file_name)
    counter = 1
    while os.path.exists(new_file_path):
        # if exists, add a counter to filename
        new_file_name = f"{subfolder_name}_{file_name}_{counter}{file_extension}"
        new_file_path = os.path.join(output_folder, new_file_name)
        counter += 1
    
    # move video to new destination with formatted filename
    shutil.move(video_path, new_file_path)
    print(f"Moved {base_name} to {new_file_path}")

if __name__ == "__main__":
    input_dir = "C:\\Users\\nicpj\\Desktop\\New folder\\AY 24-25\\temp\\datasets\\testing_named"
    output_dir = "C:\\Users\\nicpj\\Desktop\\New folder\\AY 24-25\\temp\\datasets\\lrs3_classified"

    os.makedirs(output_dir, exist_ok=True)

    process_videos(input_dir, output_dir)
