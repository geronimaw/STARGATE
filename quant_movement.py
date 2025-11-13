import numpy as np
import matplotlib.pyplot as plt

from utils import names

def main(joint_history, joint_name, fps=30):

    positions = []  # list of (x, y) tuples per frame
    frames   = []   # corresponding frame indices

    for h in joint_history:
        positions.append(h['landmarks'][joint_name])
        frames.append(h['frame'])
    
    # Convert to numpy array (with NaNs for missing)
    pts = np.array([ (x, y) if (x is not None and y is not None) else (np.nan, np.nan) 
                    for (x,y) in positions ])
    
    # Apply a simple moving average filter to smooth the trajectory
    # FIXME: filtering not working, nothing seems to change
    window_size = 50
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, axis=0, dtype=np.float64)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    pts_smoothed = np.copy(pts)
    for dim in range(2):
        valid_mask = ~np.isnan(pts[:,dim])
        valid_pts = pts[valid_mask, dim]
        if len(valid_pts) >= window_size:
            smoothed_valid = moving_average(valid_pts, n=window_size)
            pts_smoothed[valid_mask, dim][window_size-1:] = smoothed_valid

    pts_no_smooth = pts
    pts = pts_smoothed

    # Compute per-frame displacements
    dists = np.sqrt(np.nansum((pts[1:] - pts[:-1])**2, axis=1))

    total_displacement = np.nansum(dists)
    average_speed       = np.nanmean(dists) * fps  # if fps known and distances in pixels

    print(f"Total displacement for {names[joint_name]} (in pixels): {total_displacement:.2f}")
    print(f"Average speed for {names[joint_name]} (pixels/s): {average_speed:.2f}")

    # If you want angle change (for e.g., elbow = shoulder-elbow-wrist)
    # Define three joints
    joint_a = 'lm_10'
    joint_b = 'lm_11'
    joint_c = 'lm_12'

    # Collect the 3 points per frame
    pts_a = np.array([ lm.get(joint_a) if lm else (np.nan,np.nan) for lm in (h['landmarks'] for h in joint_history) ])
    pts_b = np.array([ lm.get(joint_b) if lm else (np.nan,np.nan) for lm in (h['landmarks'] for h in joint_history) ])
    pts_c = np.array([ lm.get(joint_c) if lm else (np.nan,np.nan) for lm in (h['landmarks'] for h in joint_history) ])

    # Compute angle at joint_b for each frame
    angles = []
    for (a,b,c) in zip(pts_a, pts_b, pts_c):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
            angles.append(np.nan)
        else:
            ba = a - b
            bc = c - b
            cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
            angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            angles.append(angle)

    angles = np.array(angles)
    # Range of motion
    rom = np.nanmax(angles) - np.nanmin(angles)
    print(f"Range of motion for {names[joint_name]} (degrees): {rom:.2f}")

    # # Plot joint coordinate over time
    time = np.arange(len(pts)) / fps  # in seconds
    plt.figure(figsize=(10,4))
    plt.plot(time, pts[:,0], label='x')
    plt.plot(time, pts[:,1], label='y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posizione in pixel')
    plt.title(f'Coordinata articolare nel tempo per {names[joint_name]}')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'FG4_joint_movement_{names[joint_name]}.png')
    plt.close()

    # # Plot joint speed over time
    diffs = pts[1:] - pts[:-1]
    dists = np.sqrt((diffs**2).sum(axis=1))
    time_mid = (time[1:] + time[:-1]) / 2
    plt.figure(figsize=(10,4))
    plt.plot(time_mid, dists * fps, label='Velocità (pixel/s)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocità')
    plt.title(f'Velocità articolare nel tempo per {names[joint_name]}')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'FG4_joint_speed_{names[joint_name]}.png')
    plt.close()

    # Plot trajectory in spatial plane coloured by time
    plt.figure(figsize=(6,6))
    sc = plt.scatter(pts[:,0], pts[:,1], c=time, cmap='viridis', s=5)
    plt.colorbar(sc, label='Tempo (s)')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title(f'Traiettoria articolare nello spazio per {names[joint_name]}')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'FG4_joint_trajectory_{names[joint_name]}.png')
    plt.close()

    # Plot heatmap of joint speed over time (sliding window)
    window_size = 50  # e.g., number of frames
    step = 25
    window_speeds = []
    for start in range(0, len(dists) - window_size, step):
        win = dists[start:start+window_size]
        window_speeds.append(win.mean())
    window_speeds = np.array(window_speeds)
    plt.figure(figsize=(10,2))
    plt.imshow(window_speeds[np.newaxis,:], aspect='auto', cmap='hot', 
            extent=[0, time[-1], 0, 1])
    plt.yticks([])
    plt.xlabel('Tempo (s)')
    plt.title(f'Velocità media della {names[joint_name]}')
    plt.colorbar(label='Velocità (pixel/s)')
    plt.savefig(f'FG4_joint_speed_heatmap_{names[joint_name]}.png')
    plt.close()

if __name__ == "__main__":
    # read csv file with joint_history data
    import csv
    import ast

    joint_history = []
    with open('FG4_keypoints.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['frame'])
            landmarks = {}
            for i in range(33):
                x = ast.literal_eval(row[f'lm_{i}_x'])
                y = ast.literal_eval(row[f'lm_{i}_y'])
                if x is not None and y is not None:
                    landmarks[f'lm_{i}'] = (x, y)
                else:
                    landmarks[f'lm_{i}'] = None
            joint_history.append({'frame': frame, 'landmarks': landmarks})

    joint_names = ['lm_15', 'lm_16'] # left and right wrist

    for joint_name in joint_names:
        print(f"Analysing joint: {joint_name}")
        main(joint_history, joint_name)