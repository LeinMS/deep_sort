from deep_sort_app import run_deep_sort
import os

video_list = [
    "TUD-Campus",
    "TUD-Stadtmitte",
    "KITTI-17",
    "PETS09-S2L1",
    "MOT16-09",
    "MOT16-11"
]

tracker_name = "original_deepsort"

for name in video_list:
    print(f"[Original DeepSORT] Запускаем трекинг для: {name}")

    run_deep_sort(
        sequence_dir=f"data/{name}",
        detection_npy_path=f"resources/detections/MOT16_POI_train/{name}.npy",
        output_txt=f"trackers/mot_challenge/{tracker_name}/data/{name}.txt",
        display=False
    )
