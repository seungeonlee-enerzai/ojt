import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os

def save_model(my_model, save_dir, epoch, tag=""):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = save_dir + f"/model_weight_{epoch}_epoch_{tag}.pt"
    torch.save(my_model.state_dict(), save_path)

def load_model(my_model, save_path):
    my_model.load_state_dict(torch.load(save_path))
    return my_model

def save_tensor_as_video(tensor, video_path, fps=30):
    """
    Save a (num_frame, channel, width, height) tensor as an MP4 video.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (num_frame, channel, width, height)
    video_path (str): Path to save the output video
    fps (int): Frames per second for the output video
    """
    # Ensure tensor is on CPU and convert to numpy array
    tensor = tensor.cpu().numpy()
    
    # Convert tensor shape from (num_frame, channel, width, height) to (num_frame, height, width, channel)
    tensor = np.transpose(tensor, (0, 2, 3, 1))
    
    # Ensure the pixel values are in the range [0, 255] and of type uint8
    if tensor.dtype != np.uint8:
        tensor = (tensor * 255).astype(np.uint8)
    
    num_frame, height, width, channel = tensor.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(num_frame):
        frame = tensor[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {video_path}")

def extract_patch(encoded_frame, fhd_frame, patch_size):
    """
    Extract a random patch from the given frames.
    
    Args:
    encoded_frame (torch.Tensor): The encoded frame of shape (C, H, W).
    fhd_frame (torch.Tensor): The full HD frame of shape (C, H, W).
    patch_size (tuple): The size of the patch (patch_height, patch_width).
    
    Returns:
    tuple: The patch from encoded_frame and the corresponding patch from fhd_frame.
    """
    if len(encoded_frame.shape) == 3:
        c, h, w = encoded_frame.shape
        patch_height, patch_width = patch_size
        
        if patch_height > h or patch_width > w:
            raise ValueError("Patch size should be smaller than the dimensions of the frame")
        
        y = torch.randint(0, h - patch_height + 1, (1,)).item()
        x = torch.randint(0, w - patch_width + 1, (1,)).item()
        
        encoded_patch = encoded_frame[:, y:y+patch_height, x:x+patch_width]
        fhd_patch = fhd_frame[:, y:y+patch_height, x:x+patch_width]
        
        return encoded_patch, fhd_patch
    
    else:
        n, c, h, w = encoded_frame.shape
        patch_height, patch_width = patch_size
        if patch_height > h or patch_width > w:
            raise ValueError("Patch size should be smaller than the dimensions of the frame")
        
        y = torch.randint(0, h - patch_height + 1, (1,)).item()
        x = torch.randint(0, w - patch_width + 1, (1,)).item()

        encoded_patch = encoded_frame[:, :, y:y+patch_height, x:x+patch_width]
        fhd_patch = fhd_frame[:, :, y:y+patch_height, x:x+patch_width]
        
        return encoded_patch, fhd_patch

def charbonnier_loss(pred_image, true_image, epsilon=1e-3):
    loss = torch.mean(
        (pred_image - true_image)**2 + epsilon**2
    )
    return loss

def laplacian(x):
    """
    Apply the Laplacian operator to each channel of the input tensor.
    
    Args:
    x (torch.Tensor): Input tensor of shape (B, C, H, W).
    
    Returns:
    torch.Tensor: Tensor with Laplacian applied of shape (B, C, H, W).
    """
    # Define the Laplacian kernel
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    laplacian_kernel = laplacian_kernel.repeat(x.shape[1], 1, 1, 1)  # Repeat for each channel
    
    if x.is_cuda:
        laplacian_kernel = laplacian_kernel.cuda()
    
    # Apply the Laplacian kernel to each channel
    return F.conv2d(x, laplacian_kernel, padding=1, groups=x.shape[1])

def edge_loss(pred_image, true_image, epsilon=1e-3):
    pred_edge = laplacian(pred_image.unsqueeze(0))
    target_edge = laplacian(true_image.unsqueeze(0))
    loss = nn.functional.l1_loss(pred_edge, target_edge)
    return loss

def compute_loss(pred_image, true_image):
    char_l = charbonnier_loss(pred_image, true_image)
    edge_l = edge_loss(pred_image, true_image)
    return char_l + 0.05 * edge_l


def referenced_quality_estimation(gt_video_path, test_video_path):
    """
    Estimate the quality of a video by comparing it with a ground truth video using L1 loss, SSIM, and PSNR.
    
    Args:
    gt_video_path (str): Path to the ground truth video.
    test_video_path (str): Path to the test video.
    
    Returns:
    tuple: The average L1 loss, SSIM score, and PSNR score of the video.
    """
    gt_cap = cv2.VideoCapture(gt_video_path)
    test_cap = cv2.VideoCapture(test_video_path)

    if not gt_cap.isOpened() or not test_cap.isOpened():
        raise ValueError(f"Cannot open video files {gt_video_path} or {test_video_path}")

    l1_losses = []
    ssim_scores = []
    psnr_scores = []
    
    #pdb.set_trace()
    while gt_cap.isOpened() and test_cap.isOpened():
        ret_gt, gt_frame = gt_cap.read()
        ret_test, test_frame = test_cap.read()
        
        if not ret_gt or not ret_test:
            break
       

        gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

        l1_loss = np.mean(np.abs(gt_frame.astype(np.float32) - test_frame.astype(np.float32)))
        l1_losses.append(l1_loss)

        ssim_score = ssim(gt_frame, test_frame)
        ssim_scores.append(ssim_score)
        
        psnr_score = psnr(gt_frame, test_frame)
        psnr_scores.append(psnr_score)


    gt_cap.release()
    test_cap.release()

    return np.mean(l1_losses), np.mean(ssim_scores), np.mean(psnr_scores)