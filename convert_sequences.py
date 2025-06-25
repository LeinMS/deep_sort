# convert_sequences.py

import cv2
import os
from glob import glob

def convert_sequence(img_folder, output_path, fps=30):
    img_paths = sorted(glob(os.path.join(img_folder, '*.jpg')))
    if not img_paths:
        print(f"Нет изображений в {img_folder}")
        return

    # Получим размер кадров
    sample_img = cv2.imread(img_paths[0])
    height, width = sample_img.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Не удалось создать видеофайл {output_path}")
        return

    for img_file in img_paths:
        frame = cv2.imread(img_file)
        out.write(frame)

    out.release()
    print(f"Сохранено видео: {output_path} ({len(img_paths)} кадров)")


def scan_mot_sequences(base_folder):
    seq_paths = glob(os.path.join(base_folder, '*', 'img1'))
    for img_folder in seq_paths:
        seq_name = img_folder.split(os.sep)[-2]
        output_path = os.path.join('videos', f"{seq_name}.mp4")
        convert_sequence(img_folder, output_path)

if __name__ == "__main__":
    print("Конвертация MOT16 в видео...")
    scan_mot_sequences("MOT16/test")
    scan_mot_sequences("MOT16/train")
