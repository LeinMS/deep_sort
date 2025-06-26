# run_tracker.py

import cv2
import os
import time
from detectors.yolov5_detector import YOLOv5Detector
from detectors.yolov8_detector import YOLOv8Detector  # ✅ добавлено
from detectors.fasterrcnn_detector import FasterRCNNDetector
from detectors.ssd_detector import SSDDetector

from reid_models.torchreid_model import TorchReID
from reid_models.original_deepsort import OriginalReID
from reid_models.resnet18_reid import ResNet18ReID

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from utils.drawing import draw_tracks
from utils.config_parser import load_config
from utils.save_mot_txt import save_tracking_results


def build_detector(cfg):
    name = cfg.name.lower()
    if name == "yolov5":
        return YOLOv5Detector(cfg.model_path, cfg.confidence)
    elif name == "yolov8":  # ✅ новый детектор
        return YOLOv8Detector(cfg.model_path, cfg.confidence)
    elif name == "fasterrcnn":
        return FasterRCNNDetector(cfg.confidence)
    elif name == "ssd":
        return SSDDetector(cfg.confidence)
    else:
        raise ValueError(f"Неизвестный детектор: {name}")

def build_reid(cfg):
    name = cfg.name.lower()
    if name == "torchreid":
        return TorchReID(cfg.model_name)
    elif name == "original":
        return OriginalReID(cfg.model_path)
    elif name == "resnet18":
        return ResNet18ReID(cfg.model_path)
    else:
        raise ValueError(f"Неизвестная ReID модель: {name}")

def run_tracking(video_path, output_video, output_txt, display=True, config_path="configs/deepsort_config.yaml"):
    config = load_config(config_path)

    detector = build_detector(config.detector)
    reid = build_reid(config.reid)
    metric = NearestNeighborDistanceMetric("cosine", config.tracker.max_iou_distance, config.tracker.nn_budget)
    tracker = Tracker(metric)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Не удалось создать выходной файл {output_video}")
        return

    frame_id = 1
    track_history = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections_xywh = detector.detect(frame)
        features = []
        for (x, y, w, h, conf) in detections_xywh:
            crop = frame[int(y):int(y+h), int(x):int(x+w)]
            feat = reid.extract_features(crop)
            features.append(feat)

        detections = [
            Detection([x, y, w, h], conf, feat)
            for (x, y, w, h, conf), feat in zip(detections_xywh, features)
        ]

        tracker.predict()
        tracker.update(detections)

        frame = draw_tracks(frame, tracker.tracks)

        # === Добавляем FPS и номер кадра ===
        elapsed = time.time() - start_time
        curr_fps = frame_id / elapsed
        cv2.putText(frame, f"FPS: {curr_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Frame: {frame_id}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        out.write(frame)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            tlwh = track.to_tlwh()
            track_history.append((frame_id, track.track_id, tlwh, 1.0))

        frame_id += 1

        if display:
            cv2.imshow("DeepSORT", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    save_tracking_results(output_txt, track_history)

def main():
    config = load_config("configs/deepsort_config.yaml")
    video_path = config.input.video_path
    output_video = config.input.output_video
    seq_name = os.path.splitext(os.path.basename(video_path))[0]
    output_txt = f"outputs/mot_challenge/{seq_name}/det.txt"
    run_tracking(video_path, output_video, output_txt, config.input.display)

if __name__ == "__main__":
    main()
