import os
import sys
from run_tracker import run_tracking

# Проверка аргументов командной строки
if len(sys.argv) < 2:
    print("Укажите путь к конфигурационному файлу: python run_batch.py fasterrcnn_original.yaml")
    sys.exit(1)

config_path = sys.argv[1]
tracker_name = os.path.splitext(os.path.basename(config_path))[0]  # извлекаем имя без .yaml

video_list = [
    "TUD-Campus.mp4",
    "TUD-Stadtmitte.mp4",
    "KITTI-17.mp4",
    "PETS09-S2L1.mp4",
    "MOT16-09.mp4",
    "MOT16-11.mp4"
]


tracker_name='B'
for video_file in video_list:
    name = os.path.splitext(video_file)[0]
    print(f"\n🚀 Запускаем трекинг для: {name}")

    video_path = os.path.normpath(os.path.join("videos", video_file))
    output_video = os.path.normpath(os.path.join("outputs", tracker_name, f"{name}_results.mp4"))
    output_txt = os.path.normpath(os.path.join("trackers", "mot_challenge", tracker_name, "data", f"{name}.txt"))

    run_tracking(
        video_path=video_path,
        output_video=output_video,
        output_txt=output_txt,
        display=False,
        config_path=config_path
    )
