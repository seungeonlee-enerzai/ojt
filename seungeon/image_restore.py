import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, sys

import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm
import random
from pdb import set_trace
import numpy as np
from video_restore.data import train_dataset, get_test_data
from video_restore.model import VideoRestorationCNN, cnn2, ImageRestorationModel, ComplexImageRestorationModel, VComplexImageRestorationModel, Unet
from video_restore.utils import save_model, load_model, extract_patch, compute_loss, save_tensor_as_video, referenced_quality_estimation, charbonnier_loss
from video_restore.loss import CharbonnierLoss

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from time import time


def main(args):
    print(args)

    if args.model_number == 0:
        model = VideoRestorationCNN().to(args.device)
    elif args.model_number == 1:
        model = cnn2().to(args.device)
    elif args.model_number ==2:
        model = ImageRestorationModel().to(args.device)
    elif args.model_number ==3:
        model = ComplexImageRestorationModel().to(args.device)
    elif args.model_number ==4:
        model = VComplexImageRestorationModel().to(args.device)
    elif args.model_number == 5:
        model = Unet().to(args.device)
    else:
        raise NotImplementedError("Only model 0, 1, 2, 3, 4, 5 are ready. Try again")

    if args.resume == 1:
        model = load_model(model, args.checkpoint_path)

    dataset = train_dataset(x_video_dir_path="/data1/seungeon/ojt1/x_image", y_video_dir_path="/data1/seungeon/ojt1/y_image", num_frame=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    #non_ref_videos, ref_videos = get_test_data()

    learning_rate = args.lr

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "charbonnier":
        criterion = CharbonnierLoss()
    else:
        raise NotImplementedError(f"{args.loss} function is not implemented. Select one from ['mse', 'charbonnier']")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_loss = 100

    patch_size = (args.patch_size, args.patch_size)

    for epoch_idx in range(args.num_epoch):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            #Number of image = dataset.num_frame(1) * args.batch_size (2)
            x_frames, y_frames = batch #()
            
            x_frames = torch.stack(x_frames)
            y_frames = torch.stack(y_frames)

            b, num_f, c, h, w = x_frames.shape

            x_frames = x_frames.view(-1, c, h, w)
            y_frames = y_frames.view(-1, c, h, w)


            x_frames, y_frames = extract_patch(x_frames, y_frames, patch_size)

            output = model(x_frames.cuda())

            temp_loss = criterion(output, y_frames.cuda())

            optimizer.zero_grad()
            temp_loss.backward()
            optimizer.step()

            output.detach().cpu()

            running_loss += temp_loss.item()
        
        running_loss /= len(dataloader)
        scheduler.step(running_loss)

        if running_loss < best_loss:
            best_loss = running_loss
            save_model(my_model=model, save_dir=args.save_dir, epoch=epoch_idx, tag="best")

        if epoch_idx % 50 == 0:
            save_model(my_model=model, save_dir=args.save_dir, epoch=epoch_idx, tag="log")
        print(f"Epoch {epoch_idx+1} | loss: {running_loss:.6f}")
    
    save_model(my_model=model, save_dir=args.save_dir, epoch=epoch_idx, tag="last")



    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_path', type=str, default="dataset.pt")
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--gpu_number', default="0", type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--save_dir', default="checkpoints", type=str)
    parser.add_argument('--patch', default=0, type=int)
    parser.add_argument('--model_number', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=40, type=int)
    parser.add_argument('--loss', choices=["mse", "charbonnier"], type=str, default="mse")
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--checkpoint_path', default="", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)


    main(args)