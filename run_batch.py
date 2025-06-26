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


for video_file in video_list:
    name = os.path.splitext(video_file)[0]
    print(f"\nЗапускаем трекинг для: {name}")

    run_tracking(
        video_path=f"videos/{video_file}",
        output_video=f"outputs/{tracker_name}/{name}_results.mp4",
        output_txt=f"trackers/mot_challenge/{tracker_name}/data/{name}.txt",
        display=False,
        config_path=config_path
    )
