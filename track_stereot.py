import cv2
import numpy as np

video_path     = 'F.G. 4_.mp4'
output_path    = 'F.G. 4_stereot.mp4'
fps            = None  # will be read from video

labels = [{'stereotypy': 'agita mani in aria', 'start': 0.00, 'end': 1.00},
          {'stereotypy': 'agita mani in aria', 'start': 4.00, 'end': 6.00},
          {'stereotypy': 'agita mani in aria', 'start': 10.00, 'end': 12.00}]

# Pre‐compute frames for each episode
def sec_to_frame(sec, fps):
    return int(sec * fps)

# Video capture setup
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Convert label and prediction episodes to frame intervals
label_intervals = [(sec_to_frame(l['start'], fps), sec_to_frame(l['end'], fps), l['stereotypy']) for l in labels]
frame_idx = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Example: insert your landmark drawing code here (e.g., mediapipe pose landmarks)
    #
    # e.g.:
    # results = pose.process(...)
    # mp_drawing.draw_landmarks(...)

    # Then overlay stereotypy status
    # Check if current frame is within any label interval
    for (f_start, f_end, lab) in label_intervals:
        if frame_idx >= f_start and frame_idx <= f_end:
            text = f'LABEL: {lab}'
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            # Optionally draw a colored rectangle or banner
            cv2.rectangle(frame, (0,0), (width, 40), (0,0,255), cv2.FILLED)  # red banner
            break

    # Write out frame
    out.write(frame)

    # # Optionally show live
    # cv2.imshow('Annotated Video', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done.")