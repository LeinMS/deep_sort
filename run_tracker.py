import cv2
import os
from detectors.yolov5_detector import YOLOv5Detector
from reid_models.torchreid_model import TorchReID
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from utils.config_parser import load_config
from utils.drawing import draw_tracks
from deep_sort.detection import Detection

def main():
    config = load_config("configs/deepsort_config.yaml")

    detector = YOLOv5Detector(
        model_path=config.detector.model_path,
        confidence_threshold=config.detector.confidence
    )

    reid = TorchReID(model_name=config.reid.model_name)

    metric = NearestNeighborDistanceMetric("cosine", 0.4, config.tracker.nn_budget)
    tracker = Tracker(metric)

    cap = cv2.VideoCapture(config.input.video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {config.input.video_path}")
        return

    # Подготовка VideoWriter для сохранения .mp4
    output_path = os.path.abspath(config.input.output_video)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Не удалось создать выходной файл {output_path}")
        return

    frame_id = 0

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
        out.write(frame)
        frame_id += 1

        if config.input.display:
            cv2.imshow("DeepSORT", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
