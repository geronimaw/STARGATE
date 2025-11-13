import cv2
import mediapipe as mp
import csv
from tqdm import tqdm

from utils import labels_FG4

# Pre‐compute frames for each episode
def sec_to_frame(sec, fps):
    return int(sec * fps)

def main(input_path, output_path, do_blur, do_landmark, do_stereotypy):

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Initialize face detector
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3)

    # Initialize Pose model
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,       # 0,1,2 (higher = more accurate but slower)
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Video capture setup
    cap = cv2.VideoCapture(input_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    label_intervals = [(sec_to_frame(l['start'], fps), sec_to_frame(l['end'], fps), l['stereotypy'], l['text']) for l in labels_FG4]
    frame_idx = 0
    joint_history = []  # list of dicts: {frame_idx: {landmark_name: (x,y)}}

    # Process video frame by frame
    with tqdm(total=length) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            pbar.update(1)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            face_results = face_detector.process(img_rgb)

            # Convert back to BGR for drawing
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # If face detected, apply blur to face region
            if do_blur and face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    # convert normalized bbox to pixel coords
                    x_min_bb = int(bbox.xmin * w)
                    y_min_bb = int(bbox.ymin * h)
                    bbox_w = int(bbox.width * w)
                    bbox_h = int(bbox.height * h)
                    
                    # ensure coords within image
                    x_min_bb = max(x_min_bb, 0)
                    y_min_bb = max(y_min_bb, 0)
                    x2 = min(x_min_bb + bbox_w, w)
                    y2 = min(y_min_bb + bbox_h, h)
                    
                    # extract face region
                    face_region = img_bgr[y_min_bb:y2, x_min_bb:x2]
                    # apply blur: you can tune kernel size
                    blurred_face = cv2.GaussianBlur(face_region, (71,71), 0)
                    # put blurred region back
                    img_bgr[y_min_bb:y2, x_min_bb:x2] = blurred_face
            
                # if face not detected just for few frames and landmars 15 and/or 16 overlap with previous bbox, keep previous bbox
                else:
                    if len(joint_history) > 0:
                        # last_landmarks = joint_history[-1]['landmarks']
                        # if last_landmarks is not None:
                        #     lw = last_landmarks.get('lm_15')  # left wrist
                        #     rw = last_landmarks.get('lm_16')  # right wrist
                        #     if lw is not None and rw is not None:
                        #         lw_x, lw_y = int(lw[0]), int(lw[1])
                        #         rw_x, rw_y = int(rw[0]), int(rw[1])
                        #         if (x_min_bb <= lw_x <= x2 and y_min_bb <= lw_y <= y2) or \
                        #            (x_min_bb <= rw_x <= x2 and y_min_bb <= rw_y <= y2):
                        #             # re-apply blur to previous bbox
                        face_region = img_bgr[y_min_bb:y2, x_min_bb:x2]
                        blurred_face = cv2.GaussianBlur(face_region, (91,91), 0)
                        img_bgr[y_min_bb:y2, x_min_bb:x2] = blurred_face

            # If pose landmarks detected
            if do_landmark:
                if results.pose_landmarks:
                    # Draw landmarks + connections
                    mp_drawing.draw_landmarks(
                        img_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )

                    # Extract landmark coordinates
                    landmarks = {}
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        landmarks[f"lm_{idx}"] = (lm.x * w, lm.y * h)  # pixel coords

                    joint_history.append({ 'frame': frame_idx, 'landmarks': landmarks })
                
                else:
                    joint_history.append({ 'frame': frame_idx, 'landmarks': None })

                # Plot stereotypy labels
            if do_stereotypy:
                for (f_start, f_end, lab, text) in label_intervals:
                    if frame_idx >= f_start and frame_idx <= f_end:
                        text = f'Stereotipia: {text}'
                        if lab == 1:
                            # place a rectangle around results results.pose_landmarks.landmark[11] and [12] and [23] and [24]
                            A = results.pose_landmarks.landmark[11]
                            B = results.pose_landmarks.landmark[12]
                            C = results.pose_landmarks.landmark[23]
                            D = results.pose_landmarks.landmark[24]
                            x_min = int(min(A.x, B.x, C.x, D.x) * w) - 20
                            y_min = int(min(A.y, B.y, C.y, D.y) * h) - 20
                            x_max = int(max(A.x, B.x, C.x, D.x) * w) + 20
                            y_max = int(max(A.y, B.y, C.y, D.y) * h) + 20
                            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (255,0,0), 3)
                        elif lab == 8:
                            # place a circle around wrists (results.pose_landmarks.landmark[15] and [16])
                            left_wrist = results.pose_landmarks.landmark[15]
                            right_wrist = results.pose_landmarks.landmark[16]
                            lw_x = int(left_wrist.x * w)
                            lw_y = int(left_wrist.y * h)
                            rw_x = int(right_wrist.x * w)
                            rw_y = int(right_wrist.y * h)
                            cv2.circle(img_bgr, (lw_x, lw_y), 60, (0,255,255), 3)
                            cv2.circle(img_bgr, (rw_x, rw_y), 60, (0,255,255), 3)
                        cv2.putText(img_bgr, text, (30, 60), cv2.FONT_HERSHEY_PLAIN, 4.0, (0,0,255), 2, cv2.LINE_AA)
                        break

            out.write(img_bgr)
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save landmark history to a csv file
    if do_landmark:
        with open('FG4_keypoints.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['frame'] + [f'lm_{i}_x' for i in range(33)] + [f'lm_{i}_y' for i in range(33)]
            writer.writerow(header)

            for entry in joint_history:
                frame = entry['frame']
                landmarks = entry['landmarks']
                if landmarks is not None:
                    row = [frame]
                    for i in range(33):
                        x, y = landmarks[f'lm_{i}']
                        row.extend([x, y])
                    writer.writerow(row)
                else:
                    row = [frame] + [None]*66
                    writer.writerow(row)

    # After processing: joint_history holds the per-frame landmarks
    # You can then extract trajectories, filter out noise, etc.
    print(f"Processed {frame_idx} frames")

if __name__ == "__main__":
    blur = True
    landmark = False
    stereotypy = False

    # Video input & output
    input_path  = 'FG4_.mp4'
    output_path = f"FG4_{'blur' if blur else ''}{'_lm' if landmark else ''}{'_stereo' if stereotypy else ''}.mp4"
    main(input_path, output_path, blur, landmark, stereotypy)