import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

'''
Classify angle based on y-axis rotation.

Parameters:
    y (int): TODO
Returns:
    angle (str): TODO
'''
def classify_angle(y):    
    # # NOTE: TEMPORARY THRESHOLDS
    # # 90 DEGREES (SIDE)
    # if y < -22 or y > 22:
    #     angle = "side"
    # # 60 DEGREES
    # elif y < -18 or y > 18:
    #     angle = "60"
    # # 45 DEGREES
    # elif y < -14 or y > 14:
    #     angle = "45"
    # # 30 DEGREES
    # elif y < -8 or y > 8:
    #     angle = "30"
    # else:
    #     angle = "front"

    # return angle

    # Front: -15 to 15 degrees
    if -15 <= y <= 15:
        return "front"
    # Side: Extreme angles (>60 or <-60 degrees)
    elif y > 60 or y < -60:
        return "side"
    # 60 degrees: More precise range for angles around 60
    elif -60 < y <= -45 or 45 <= y < 60:
        return "60"
    # 45 degrees: More precise range for angles around 45
    elif -45 < y <= -30 or 30 <= y < 45:
        return "45"
    # 30 degrees: More precise range for angles around 30
    elif -30 < y <= -15 or 15 < y < 30:
        return "30"
    else:
        return "unknown"  # For angles that do not fit the expected range

'''
Process all videos in a folder.

Parameters:
    input_folder (str): TODO
    output_folder (str): TODO
'''
def process_videos(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for video_name in files:
            if video_name.endswith((".mp4")):
                video_path = os.path.join(root, video_name)
                classify_video(video_path, output_folder)


'''
Process a single video.

Parameters:
    video_path (str): TODO
    output_folder (str): TODO
'''
def classify_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    angle_counts = {
        "front": 0,
        "side": 0,
        "30": 0,
        "45": 0,
        "60": 0
    }

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # preprocess frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

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

                        y_angle = angles[1] * 360
                        detected_angle = classify_angle(y_angle)
                        angle_counts[detected_angle] += 1

    cap.release()

    # classify based on angle counts
    angles_above_threshold = [angle for angle, count in angle_counts.items() if count >= 10]

    if len(angles_above_threshold) > 1:
        classification = "mixed"
    else:
        classification = max(angle_counts, key=angle_counts.get)

    # move video to classified folder
    output_subfolder = os.path.join(output_folder, classification)
    os.makedirs(output_subfolder, exist_ok=True)
    shutil.move(video_path, os.path.join(output_subfolder, os.path.basename(video_path)))

    print(f"Classified {os.path.basename(video_path)} as {classification}")

if __name__ == "__main__":
    input_dir = "C:\\Users\\nicpj\Desktop\\New folder\\AY 24-25\\temp\\datasets\\lrs3_test_v0.4"
    output_dir = "C:\\Users\\nicpj\Desktop\\New folder\\AY 24-25\\temp\\datasets\\lrs3_classified"

    os.makedirs(output_dir, exist_ok=True)

    process_videos(input_dir, output_dir)
