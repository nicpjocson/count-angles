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

angle_counts = {
    0: 0, # front
    15: 0, 
    30: 0, 
    45: 0, 
    60: 0, 
    90: 0 # side
}

for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".mp4"):
            print(filename)
            cap = cv2.VideoCapture(os.path.join(directory, filename))
            
            # FIX: NOT ENTERING THIS LOOP
            while cap.isOpened():
                print("0000")
                success, image = cap.read()
                print("1111")
                if not success:
                    print("2222")
                    break
                else:
                    print("3333")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True

                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    print("test")
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            # TODO: can use more points
                            if idx in [33, 263, 1, 61, 291, 199]:
                                # if idx == 1:
                                #     nose_2d = (lm.x * img_w, lm.y * img_h)
                                #     nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w
                        cam_matrix = np.array([[focal_length, 0, img_h / 2], 
                                            [0, focal_length, img_w / 2], 
                                            [0, 0, 1]])
                    
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # x = angles[0] * 360
                        y = angles[1] * 360
                        # z = angles[2] * 360

                        # see where head is tilting
                        # TEMPORARY
                        if y < -60:
                            angle_counts[90] += 1
                        elif y < -45:
                            angle_counts[60] += 1
                        elif y < -30:
                            angle_counts[45] += 1
                        elif y < -15:
                            angle_counts[30] += 1
                        elif y < -5:
                            angle_counts[15] += 1
                        else:
                            angle_counts[0] += 1

                        # display nose direction
                        # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                        # cv2.line(image, p1, p2, (255, 0, 0), 3)

                        # add text to image
                        # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        # cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # cv2.putText(image, "y: " + str(np.round(x,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # cv2.putText(image, "z: " + str(np.round(x,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # end = time.time()
                    # total_time = end - start
                    # fps = 1 / total_time
                    # print("FPS: ", fps)
                    # cv2.putText(image, f'FPS: {int(fps)}' , (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                    # draw landmarks with drawing utilities set up
                    # mp_drawing.draw_landmarks(
                    #     image=image, 
                    #     landmark_list=face_landmarks, 
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS, 
                    #     landmark_drawing_spec=drawing_spec, 
                    #     connection_drawing_spec=drawing_spec
                    # )

        # cv2.imshow("Head Pose Estimation", image)

        # `esc` or `q` key to terminate program
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

            cap.release()

with open("head_pose_results.txt", "w") as f:
    f.write(f"0 (front) {angle_counts[0]}\n")
    f.write(f"15 {angle_counts[15]}\n")
    f.write(f"30 {angle_counts[30]}\n")
    f.write(f"45 {angle_counts[45]}\n")
    f.write(f"60 {angle_counts[60]}\n")
    f.write(f"90 (side) {angle_counts[90]}\n")

