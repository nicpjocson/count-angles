import cv2
import mediapipe as mp
import numpy as np
import os
# import time

directory = "D:\\New Folder\\thesis\\datasets\\testing" # CHANGE IF NEEDED

mp_face_mesh = mp.solutions.face_mesh # open face mesh detector
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils # for displaying whole face mesh
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# while webcam is open
while cap.isOpened():
    # read image from webcam
    success, image = cap.read()

    # flip image horizontally for a later selfie-view display (i.e. not mirrored/flipped)
    # convert color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # for improving performance
    # makes it so program can only read from the image
    image.flags.writeable = False

    # get result (returns normalized values)
    results = face_mesh.process(image)

    # for improving performance
    # can write on image (i.e. display text, etc)
    image.flags.writeable = True

    # image height, width, and number of channels
    # for scaling values with image dimensions
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    # if there are detections
    if results.multi_face_landmarks:
        # run thru all landmarks detected in the image
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # indexes for e.g. nose, ears, mouth, eyes
                # TODO: can use more points
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    # coordinates of landmark (normalized values)
                    # scale with height and width of image (i.e. convert back to image space)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # get 2D coordinates
                    face_2d.append([x, y])
                    # get 3D coordinates
                    face_3d.append([x, y, lm.z])

            # convert 2D coordinates to numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            # covert 3D coordinates to numpy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # set up camera matrix
            focal_length = 1 * img_w

            # intrinsic parameters of camera
            cam_matrix = np.array([[focal_length, 0, img_h / 2], 
                                   [0, focal_length, img_w / 2], 
                                   [0, 0, 1]])
for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".mp4"):
            print(filename)
            cap = cv2.VideoCapture(os.path.join(directory, filename))
            
            # distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # solve PnP
            # rot_vec = how much the points are rotated in the image
            # trans_vec = how much the points are translated around
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # get angles for all axes (normalized)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # see where head is tilting
            # TODO
            # count angle
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # add text to image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(x,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(x,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # TODO
            # add count to txt

        # draw landmarks with drawing utilities set up
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=face_landmarks, 
            connections=mp_face_mesh.FACEMESH_CONTOURS, 
            landmark_drawing_spec=drawing_spec, 
            connection_drawing_spec=drawing_spec
        )

    cv2.imshow("Head Pose Estimation", image)

    # `esc` or `q` key to terminate program
    if cv2.waitKey(5) & 0xFF == 27:
        break

# release webcam
cap.release()
                        # display nose direction
                        # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                        # cv2.line(image, p1, p2, (255, 0, 0), 3)

                    # end = time.time()
                    # total_time = end - start
                    # fps = 1 / total_time
                    # print("FPS: ", fps)
                    # cv2.putText(image, f'FPS: {int(fps)}' , (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

with open("head_pose_results.txt", "w") as f:
    f.write(f"0 (front) {angle_counts[0]}\n")
    f.write(f"15 {angle_counts[15]}\n")
    f.write(f"30 {angle_counts[30]}\n")
    f.write(f"45 {angle_counts[45]}\n")
    f.write(f"60 {angle_counts[60]}\n")
    f.write(f"90 (side) {angle_counts[90]}\n")

