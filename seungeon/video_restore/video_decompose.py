import os, sys

import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
    cap.release()
    return frames

def decompose_video(video_name, video_dir_path, save_dir_path):
    video_path = os.path.join(video_dir_path, video_name)
    video_name = os.path.basename(video_path).replace(".mp4", "").replace(".y4m", "")
    save_video_dir_name = os.path.join(save_dir_path, video_name)

    if not os.path.exists(save_video_dir_name):
        os.mkdir(save_video_dir_name)
    else:
        return
    
    video_frames = load_video_frames(video_path)
    print(video_path)
    
    for idx, video_frame in enumerate(video_frames):
        final_save_img_name = os.path.join(save_video_dir_name, f"{idx}.png")
        cv2.imwrite(final_save_img_name, video_frame)

"""
if __name__ == "__main__":
    train_x_dir_path = "../train/encoded_full"
    train_y_dir_path = "../train/FHD"

    save_x_dir_path = "../train/x_image"
    save_y_dir_path = "../train/y_image"

    train_x_names = os.listdir(train_x_dir_path)
    train_y_names = os.listdir(train_y_dir_path)

    
    for train_x_name in train_x_names:
        train_x_path = os.path.join(train_x_dir_path, train_x_name)
        video_name = os.path.basename(train_x_path).replace(".mp4", "")
        save_video_dir_name = os.path.join(save_x_dir_path, video_name)

        if not os.path.exists(save_video_dir_name):
            os.mkdir(save_video_dir_name)

        video_frames = load_video_frames(train_x_path)

        print(train_x_path)
        for idx, video_frame in enumerate(video_frames):
            final_save_img_name = os.path.join(save_video_dir_name, f"{idx}.png")
            cv2.imwrite(final_save_img_name, video_frame)
    
    for train_y_name in train_y_names:
        train_y_path = os.path.join(train_y_dir_path, train_y_name)
        
        video_name = os.path.basename(train_y_path).replace(".mp4", "")

        save_video_dir_name = os.path.join(save_y_dir_path, video_name)

        if not os.path.exists(save_video_dir_name):
            os.mkdir(save_video_dir_name)

        video_frames = load_video_frames(train_y_path)

        print(train_y_path)
        for idx, video_frame in enumerate(video_frames):
            final_save_img_name = os.path.join(save_video_dir_name, f"{idx}.png")
            cv2.imwrite(final_save_img_name, video_frame)
"""

import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed



if __name__ == "__main__":
    
    train_x_dir_path = "../train/encoded_full"
    train_y_dir_path = "../train/FHD"

    save_x_dir_path = "/data1/seungeon/ojt1/x_image"
    save_y_dir_path = "/data1/seungeon/ojt1/y_image"

    #train_x_names = sorted([f for f in os.listdir(train_x_dir_path) if f.endswith("_40.mp4")])
    train_x_names = sorted(os.listdir(train_x_dir_path))
    train_y_names = sorted(os.listdir(train_y_dir_path))
    

    max_workers=30

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(decompose_video, train_x_name, train_x_dir_path, save_x_dir_path)

            for train_x_name in train_x_names
        ]
        for future in as_completed(futures):
            future.result()  # Handle exceptions if any
    sys.exit("Program end")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(decompose_video, train_y_name, train_y_dir_path, save_y_dir_path)
            for train_y_name in train_y_names
        ]
        for future in as_completed(futures):
            future.result()  # Handle exceptions if any
    """
    test_x_dir_path = "../test"
    test_y_dir_path = "../test/GT"

    save_x_dir_path = "/data1/seungeon/ojt1/test_x_image"
    save_y_dir_path = "/data1/seungeon/ojt1/test_y_image"

    test_x_names = sorted([f for f in os.listdir(test_x_dir_path) if f.endswith(".mp4")])
    test_y_names = sorted([f for f in os.listdir(test_y_dir_path) if f.endswith(".y4m")])

    max_workers=30

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(decompose_video, train_x_name, test_x_dir_path, save_x_dir_path)

            for train_x_name in test_x_names
        ]
        for future in as_completed(futures):
            future.result()  # Handle exceptions if any

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(decompose_video, train_y_name, test_y_dir_path, save_y_dir_path)
            for train_y_name in test_y_names
        ]
        for future in as_completed(futures):
            future.result()  # Handle exceptions if any

    """