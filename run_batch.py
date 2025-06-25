from run_tracker import run_tracking
import os

video_list = [
    "TUD-Campus.mp4",
    "TUD-Stadtmitte.mp4",
    "KITTI-17.mp4",
    "PETS09-S2L1.mp4",
    "MOT16-09.mp4",
    "MOT16-11.mp4"
]

tracker_name = "yolov5_torchreid_tracker"

for video_file in video_list:
    name = os.path.splitext(video_file)[0]  # без .mp4
    print(f"\n🚀 Запускаем трекинг для: {name}")

    run_tracking(
        video_path=f"videos/{video_file}",
        output_video=f"outputs/{name}_results.mp4",
        output_txt=f"trackers/mot_challenge/{tracker_name}/data/{name}.txt",
        display=False
    )
