import cv2
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
from tqdm import tqdm
import subprocess
import random
from multiprocessing import Pool, Process, Queue
from pdb import set_trace

def encode_video(origin_video_path, crf_factor, outfile_path):
   
    if not os.path.exists(outfile_path):
        subprocess.run(['ffmpeg', '-i', origin_video_path,
                                '-c:v', 'libx264',
                                '-c:a', 'copy',
                                '-crf', str(crf_factor),
                                outfile_path,
                                '-loglevel', 'quiet',
                                '-y'
                        ])
    else:
        return


DATA_DIR_PATH = "./"

train_img_dir_path = os.path.join(DATA_DIR_PATH, 'train', 'FHD')
test_img_dir_path = os.path.join(DATA_DIR_PATH, 'test')

train_video_names = os.listdir(train_img_dir_path)
train_video_names.sort()

crf_start = 24
crf_end  = 29
fps = 5
num_train = len(train_video_names)

print("start")

"""
result_queue = Queue()
max_process_number=40
#Multiprocessing
train_video_path_list = [os.path.join("./train/FHD", video_name) for video_name in train_video_names]
crf_factor_list = [random.randint(crf_start, crf_end) for _ in range(len(train_video_names))]
outfile_path_list = [os.path.join("./train/encoded_full", video_name.replace(".mp4", f"_{crf_factor_list[video_idx]}.mp4")) for video_idx, video_name in enumerate(train_video_names)]

process_list = [Process(target=encode_video, args=(train_video_path, crf_factor_list[video_idx], outfile_path_list[video_idx])) for video_idx, train_video_path in enumerate(train_video_path_list)]

for process in process_list:
    process.start()

for process in process_list:
    process.join()
"""


#Iterative Code
for idx, train_video_name in tqdm(enumerate(train_video_names)):
    print(f"\r{idx}/{num_train}", end="  ")
    train_video_path = os.path.join("./train/FHD", train_video_name)
    #ffmpeg -i ${video_name} -c:v libx264 -preset ${preset} -crf 1 -profile:v ${profile} results/output_${preset}_1_${profile}.mp4

    crf_factor = random.randint(crf_start, crf_end)

    
    outfile_name = train_video_name.replace(".mp4", f"_{crf_factor}.mp4")
    outfile_path = os.path.join("./train/encoded_full", outfile_name)

    print(outfile_path)
    if not os.path.exists(outfile_path):
        #print(train_video_path, cfr, outfile_path)
        subprocess.run(['ffmpeg', '-i', train_video_path,
                                '-c:v', 'libx264',
                                '-c:a', 'copy',
                                '-crf', str(crf_factor),
                                outfile_path,
                                '-loglevel', 'quiet',
                                '-y'
                        ])
