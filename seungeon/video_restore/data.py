import torch
import os, sys

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
from pdb import set_trace

from concurrent.futures import ThreadPoolExecutor, as_completed

import random


def sample_same_indices(list1, list2, n):
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length.")
    
    if n > len(list1):
        raise ValueError("n must be less than or equal to the length of the lists.")
    
    # 리스트의 인덱스를 섞고 상위 n개의 인덱스를 선택

    sampled_indices = random.sample(range(len(list1)), n)
    
    # 선택된 인덱스에 위치한 요소들을 반환
    sampled_list1 = [list1[i] for i in sampled_indices]
    sampled_list2 = [list2[i] for i in sampled_indices]
    
    return sampled_list1, sampled_list2

"""
class VideoRestorationDataset(Dataset):
    def __init__(self, args, fhd_dir, encoded_dir, transform=None):
        self.fhd_dir = fhd_dir
        self.encoded_dir = encoded_dir
        self.transform = transform
        self.args = args
        self.video_pairs = self._create_pairs()
        self.load_videos()
        

    def _create_pairs(self):
        fhd_files = [f for f in os.listdir(self.fhd_dir) if f.endswith('.mp4')]
        fhd_files = random.sample(fhd_files, self.args.num_sample//2)
        video_pairs = []

        for fhd_file in fhd_files:
            base_name = os.path.splitext(fhd_file)[0]
            for suffix in ['_40', '_51']:
                encoded_file = f"{base_name}{suffix}.mp4"
                if os.path.exists(os.path.join(self.encoded_dir, encoded_file)):
                    video_pairs.append((fhd_file, encoded_file))

        return video_pairs

    def __len__(self):
        return len(self.video_pairs)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames
    
    def load_videos(self):
        self.encoded_frame_list = []
        self.fhd_frame_list = []

        print("Loading Videos")
        for idx in tqdm(range(len(self.video_pairs))):
            fhd_file, encoded_file = self.video_pairs[idx]
            fhd_path = os.path.join(self.fhd_dir, fhd_file)
            encoded_path = os.path.join(self.encoded_dir, encoded_file)

            fhd_frames = self._load_video_frames(fhd_path)
            encoded_frames = self._load_video_frames(encoded_path)

            if self.transform:
                fhd_frames = [self.transform(frame) for frame in fhd_frames]
                encoded_frames = [self.transform(frame) for frame in encoded_frames]

            fhd_frames = torch.stack(fhd_frames)
            encoded_frames = torch.stack(encoded_frames)

            self.fhd_frame_list.append(fhd_frames)
            self.encoded_frame_list.append(encoded_frames)


    def __getitem__(self, idx):
        encoded_frames = self.encoded_frame_list[idx]
        fhd_frames = self.fhd_frame_list[idx]

        return encoded_frames, fhd_frames
"""

class VideoRestorationDataset(Dataset):
    def __init__(self, args, fhd_dir, encoded_dir, transform=None):
        self.fhd_dir = fhd_dir
        self.encoded_dir = encoded_dir
        self.transform = transform
        self.args = args
        self.video_pairs = self._create_pairs()
        

    def _create_pairs(self):
        fhd_files = [f for f in os.listdir(self.fhd_dir) if f.endswith('.mp4')]
        fhd_files = random.sample(fhd_files, self.args.num_sample//2)
        video_pairs = []

        for fhd_file in fhd_files:
            base_name = os.path.splitext(fhd_file)[0]
            for suffix in ['_40', '_51']:
                encoded_file = f"{base_name}{suffix}.mp4"
                if os.path.exists(os.path.join(self.encoded_dir, encoded_file)):
                    video_pairs.append((fhd_file, encoded_file))

        return video_pairs

    def __len__(self):
        return len(self.video_pairs)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        fhd_file, encoded_file = self.video_pairs[idx]
        fhd_path = os.path.join(self.fhd_dir, fhd_file)
        encoded_path = os.path.join(self.encoded_dir, encoded_file)

        fhd_frames = self._load_video_frames(fhd_path)
        encoded_frames = self._load_video_frames(encoded_path)

        if self.transform:
            fhd_frames = [self.transform(frame) for frame in fhd_frames]
            encoded_frames = [self.transform(frame) for frame in encoded_frames]

        fhd_frames = torch.stack(fhd_frames)
        encoded_frames = torch.stack(encoded_frames)

        return encoded_frames, fhd_frames

class inference_dataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.fhd_dir = video_dir
        self.transform = transform

        self.videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.videos.sort()

    def __len__(self):
        return len(self.videos)
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        video_name = os.path.basename(video_path)

        return frames, fps, video_name


    def __getitem__(self, idx):
        video_path = os.path.join(self.fhd_dir, self.videos[idx])

        fhd_frames, fps, video_name = self._load_video_frames(video_path)

        if self.transform:
            fhd_frames = [self.transform(frame) for frame in fhd_frames]

        fhd_frames = torch.stack(fhd_frames)

        return fhd_frames, fps, video_name


def load_img(img_path, save_list, save_idx):
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = np.transpose(img, (2, 0, 1))
    
    #img = torch.Tensor(img)
    #save_list[save_idx] = img

    img = np.array(Image.open(img_path))
    img = np.transpose(img, (2, 0, 1))/255.0
    img = img.astype(np.float32)
    img = torch.tensor(img, dtype=torch.float32)
    
    save_list[save_idx] = img

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)
        frames.append(frame)
    cap.release()
    return frames

def get_test_data():
    test_video_dir_path = "/data1/seungeon/ojt1"

    non_ref_video_names = ["CoverSong_720P-7360_crf_10_ss_00_t_20.0", "Lecture_1080P-011f_ss_00_t_20.0_INGEST", "LiveMusic_360P-1d94_crf_10_ss_00_t_20.0", "Sniper_INGEST"]
    ref_video_names = ["MeridianTalk_INGEST", "TunnelFlag_INGEST"]
    
    non_ref_videos = {}
    for name in non_ref_video_names:
        non_ref_videos[name] = [0 for _ in range(len(os.listdir(os.path.join(test_video_dir_path,"test_x_image", name))))]

        for idx, frame in enumerate(os.listdir(os.path.join(test_video_dir_path,"test_x_image", name))):
            load_img(os.path.join(test_video_dir_path,"test_x_image", name, str(idx)+'.png'), non_ref_videos[name], idx)

    ref_videos = {}
    for name in ref_video_names:
        ref_videos[name] = [0 for _ in range(len(os.listdir(os.path.join(test_video_dir_path, "test_y_image", name))))]

        for idx, frame in enumerate(os.listdir(os.path.join(test_video_dir_path,"test_y_image", name))):
            load_img(os.path.join(test_video_dir_path,"test_y_image", name, str(idx)+'.png'), ref_videos[name], idx)

            
    return non_ref_videos, ref_videos

def get_train_video_names(train_video_dir_path):
    video_names = [f for f in os.listdir(train_video_dir_path) if not f.endswith("_51")]
    random.shuffle(video_names)
    return video_names

class train_dataset(Dataset):
    def __init__(self, x_video_dir_path, y_video_dir_path, num_frame=3):
        self.x_video_dir = x_video_dir_path
        self.y_video_dir = y_video_dir_path

        self.x_videos = [f for f in os.listdir(x_video_dir_path)]
        self.y_videos = [f for f in os.listdir(y_video_dir_path)]

        self.x_videos.sort()
        self.y_videos.sort()

        self.x_video_names = get_train_video_names(x_video_dir_path)
        self.length = len(self.x_video_names)

        self.num_frame = num_frame

    def __len__(self):
        return self.length
    
    def find_y_for_x(self, x_video_name):
        video_name = x_video_name.split("_")[0]
        return video_name
    
    def __getitem__(self, idx):
        #set_trace()
        try:
            x_video_path = os.path.join(self.x_video_dir, self.x_video_names[idx])
            x_frame_names = os.listdir(x_video_path)

            y_video_path = os.path.join(self.y_video_dir, self.find_y_for_x(self.x_video_names[idx]))
            y_frame_names = os.listdir(y_video_path)

            x_frame_names, y_frame_names = sample_same_indices(x_frame_names, y_frame_names, n=self.num_frame)

            x_frame_list = [0 for _ in range(len(x_frame_names))]
            y_frame_list = [0 for _ in range(len(y_frame_names))]
        except:
            return self.__getitem__(idx+1)

        #max_workers=self.num_frame
        
        for idx, frame_name in enumerate(x_frame_names):
            load_img(os.path.join(x_video_path, frame_name), x_frame_list, idx)

        for idx, frame_name in enumerate(y_frame_names):
            load_img(os.path.join(y_video_path, frame_name), y_frame_list, idx)
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(load_img, os.path.join(x_video_path, frame_name), x_frame_list, idx)

                for idx, frame_name in enumerate(x_frame_names)
            ]
            for future in as_completed(futures):
                future.result()  # Handle exceptions if any


        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(load_img, os.path.join(y_video_path, frame_name), y_frame_list, idx)

                for idx, frame_name in enumerate(y_frame_names)
            ]
            for future in as_completed(futures):
                future.result()  # Handle exceptions if any
        """

        """
        for frame_name in x_frame_names:
            frame_path = os.path.join(x_video_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            
            img = torch.Tensor(img)

            x_frame_list.append(img)
        
        for frame_name in y_frame_names:
            frame_path = os.path.join(y_video_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = torch.Tensor(img)

            y_frame_list.append(img)
        """
        
        return x_frame_list, y_frame_list