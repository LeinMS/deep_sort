# save_mot_txt.py
# Сохраняет треки DeepSORT в формате MOTChallenge для последующей оценки (TrackEval)

import os

def save_tracking_results(output_path, track_history):
    """
    Сохраняет список треков в файл MOTChallenge формата.
    
    output_path: str — путь до .txt файла
    track_history: List[Tuple[frame_id, track_id, [x, y, w, h], score]]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for frame_id, track_id, tlwh, score in track_history:
            x, y, w, h = tlwh
            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1\n"
            f.write(line)

    print(f"✅ Saved tracking results to: {output_path}")


if __name__ == "__main__":
    dummy_tracks = [
        (1, 1, [100, 50, 80, 200], 1.0),
        (1, 2, [400, 60, 90, 190], 1.0),
        (2, 1, [102, 53, 80, 200], 1.0),
    ]
    save_tracking_results("trackers/mot_challenge/MOT16-01/ours/det.txt", dummy_tracks)
