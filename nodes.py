import torch
import numpy as np
import json
import cv2
from torchvision.transforms.functional import resize as torch_resize
from torch.nn.functional import interpolate
import os
from PIL import Image
import folder_paths
import time
import librosa
import math
import io
import pyaudio
import wave
from pydub import AudioSegment

from typing import List, Tuple, Optional
import audioop
import tempfile
from comfy import model_management
import audioop
import tempfile
import torchaudio
from torchaudio.transforms import Fade


class ImageAreaCropNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "do_resize": ("BOOLEAN", {"default": False}),
                "scaled_width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "scaled_height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON",)
    RETURN_NAMES = ("image", "crop_info",)
    FUNCTION = "crop_image"
    CATEGORY = "image"

    def crop_image(self, image, x, y, width, height, do_resize, scaled_width, scaled_height):
        # 确保在GPU上处理
        device = image.device
        B, H, W, C = image.shape
        
        # 批量边界修正
        x = torch.clamp(torch.tensor(x, device=device), min=0, max=W-1).item()
        y = torch.clamp(torch.tensor(y, device=device), min=0, max=H-1).item()
        width = min(width, W - x)
        height = min(height, H - y)
        
        # 批量裁剪操作
        cropped = image[:, y:y+height, x:x+width, :]
        
        if do_resize:
            # 使用更高效的批量resize (torch.nn.functional.interpolate)
            # 调整维度顺序 (B, H, W, C) -> (B, C, H, W)
            cropped = cropped.permute(0, 3, 1, 2)
            
            # 使用GPU加速的插值方法
            cropped = interpolate(
                cropped, 
                size=(scaled_height, scaled_width), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 恢复维度顺序 (B, C, H, W) -> (B, H, W, C)
            cropped = cropped.permute(0, 2, 3, 1)
        
        # 批量裁剪信息
        crop_info = {
            "original_size": [W, H],
            "crop_position": [x, y],
            "crop_size": [width, height],
            "do_resize": do_resize,
            "scaled_size": [scaled_width, scaled_height] if do_resize else None
        }
        crop_info_list = [crop_info] * B
        
        return (cropped, crop_info_list,)


class AreaCropRestoreNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "crop_info": ("JSON",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore_image"
    CATEGORY = "image"

    def restore_image(self, cropped_image, target_image, crop_info):
        device = cropped_image.device
        B, H, W, C = target_image.shape
        
        # 复制目标图像以避免原地修改
        restored = target_image.clone()
        
        # 确保裁切信息是列表格式
        if not isinstance(crop_info, list):
            crop_info = [crop_info] * B
        
        # 为批量处理准备坐标和尺寸
        x_coords = []
        y_coords = []
        target_heights = []
        target_widths = []
        
        for i in range(B):
            info = crop_info[i] if i < len(crop_info) else crop_info[0]
            x, y = info["crop_position"]
            w, h = info["crop_size"]
            
            # 修正边界
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = min(w, W - x)
            h = min(h, H - y)
            
            x_coords.append(x)
            y_coords.append(y)
            target_widths.append(w)
            target_heights.append(h)
        
        # 创建一个列表来存储调整后的裁剪图像
        resized_crops = []
        
        # 分组处理相同尺寸的图像以提高效率
        size_groups = {}
        for i in range(B):
            size_key = (target_heights[i], target_widths[i])
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append(i)
        
        # 按尺寸组处理
        for (target_h, target_w), indices in size_groups.items():
            # 获取该组的所有裁剪图像
            group_crops = cropped_image[indices]
            
            # 检查是否需要调整尺寸
            if group_crops.shape[1] != target_h or group_crops.shape[2] != target_w:
                # 调整维度顺序 (B, H, W, C) -> (B, C, H, W)
                group_crops = group_crops.permute(0, 3, 1, 2)
                
                # 批量调整到目标尺寸
                resized_group = interpolate(
                    group_crops,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # 恢复维度顺序 (B, C, H, W) -> (B, H, W, C)
                resized_group = resized_group.permute(0, 2, 3, 1)
            else:
                resized_group = group_crops
            
            # 存储调整后的图像
            for i, idx in enumerate(indices):
                resized_crops.append((idx, resized_group[i]))
        
        # 按原始索引排序
        resized_crops.sort(key=lambda x: x[0])
        resized_crops = [crop for _, crop in resized_crops]
        
        # 将图像贴回目标位置
        for i in range(B):
            # 获取当前裁剪图像
            current_crop = resized_crops[i]
            
            # 确保尺寸匹配
            if current_crop.shape[0] != target_heights[i] or current_crop.shape[1] != target_widths[i]:
                # 如果尺寸仍不匹配，强制调整
                current_crop = current_crop.permute(2, 0, 1).unsqueeze(0)
                current_crop = interpolate(
                    current_crop,
                    size=(target_heights[i], target_widths[i]),
                    mode='bilinear',
                    align_corners=False
                )
                current_crop = current_crop.squeeze(0).permute(1, 2, 0)
            
            # 替换目标图像中的区域
            restored[i, y_coords[i]:y_coords[i]+target_heights[i], 
                    x_coords[i]:x_coords[i]+target_widths[i], :] = current_crop
        
        return (restored,)


class ImageReverseOrderNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reverse": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": False}),
                "loop_count": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 100,
                    "step": 1
                }),
                "deduplicate": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_sequence"
    CATEGORY = "image"
    
    def process_sequence(self, images, reverse, loop, loop_count, deduplicate):
        # 获取设备信息 (CPU/GPU)
        device = images.device
        n = images.size(0)
        
        # 非循环模式 - 直接处理
        if not loop:
            return (images.flip([0]) if reverse else images,)
        
        # 预计算索引
        # 创建完整正序和倒序索引
        forward_indices = torch.arange(n, device=device)
        backward_indices = torch.flip(forward_indices, [0])
        
        # 创建去尾索引
        dedup_forward = forward_indices[:-1] if deduplicate and n > 1 else forward_indices
        dedup_backward = backward_indices[:-1] if deduplicate and n > 1 else backward_indices
        
        # 确定序列构建方向
        segments = []
        
        # 添加起始段
        if reverse:
            segments.append(backward_indices)
        else:
            segments.append(forward_indices)
        
        # 添加循环段
        for i in range(loop_count):
            # 添加相反方向的段
            if reverse:
                segments.append(dedup_forward)
            else:
                segments.append(dedup_backward)
            
            # 添加相同方向的段
            if reverse:
                segments.append(dedup_backward)
            else:
                segments.append(dedup_forward)
        
        # 合并所有段
        combined_indices = torch.cat(segments)
        
        # 使用索引选择代替数据复制
        processed_images = images.index_select(0, combined_indices)
        
        return (processed_images,)
    


class ImageTransferNode:
    def __init__(self):
        self.cached_images = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_manual": ("BOOLEAN", {"default": False}),
                "merge_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 9999,
                    "step": 1
                }),
            },
            "optional": {
                "image": ("IMAGE",),  # Optional input to update cache
                "manual_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transfer_images"
    CATEGORY = "image"
    
    def transfer_images(self, merge_manual, merge_index, image=None, manual_image=None):
        # Update cache if new image is provided
        if image is not None and image.numel() > 0:
            self.cached_images = image.clone()
        
        # Determine output: use cache or manual image
        if self.cached_images is not None:
            output_images = self.cached_images
        elif manual_image is not None:
            output_images = manual_image
        else:
            # Return empty tensor if no images available
            return (torch.zeros(0, 64, 64, 3),)
        
        # Handle manual image merging
        if merge_manual and manual_image is not None:
            n = output_images.size(0)
            index = min(max(merge_index, 0), n)  # Clamp index to valid range
            
            # Efficient insertion using torch.cat
            if index == 0:
                output_images = torch.cat([manual_image, output_images], dim=0)
            elif index >= n:
                output_images = torch.cat([output_images, manual_image], dim=0)
            else:
                output_images = torch.cat([
                    output_images[:index],
                    manual_image,
                    output_images[index:]
                ], dim=0)
        
        return (output_images,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always consider node as changed



class AudioSplitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("INT", {"default": 25, "min": 1}),
                "split_length": ("INT", {"default": 25, "min": 1}),
                "skip_length": ("INT", {"default": 0, "min": 0}),
                "transition_length": ("INT", {"default": 0, "min": 0}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("segmented_audio",)
    FUNCTION = "split_audio"
    CATEGORY = "audio/utils"

    def split_audio(self, audio, frame_rate, split_length, skip_length, transition_length):
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 确保波形是3D张量 (batch, channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
        
        batch_size, num_channels, total_samples = waveform.shape
        
        # 计算总音频时长（毫秒）
        total_duration_ms = (total_samples / sample_rate) * 1000
        
        # 计算帧时长（毫秒/帧）
        ms_per_frame = 1000 / frame_rate
        
        # 计算跳过位置和分割区间
        skip_ms = skip_length * ms_per_frame
        transition_ms = transition_length * ms_per_frame
        
        # 实际起始位置（应用过渡偏移）
        start_ms = max(0, skip_ms - transition_ms)
        end_ms = start_ms + (split_length + transition_length) * ms_per_frame
        
        # 处理边界情况
        if skip_ms > total_duration_ms:
            start_ms = max(0, total_duration_ms - split_length * ms_per_frame)
            end_ms = total_duration_ms
            transition_ms = 0  # 末尾取消过渡效果
        elif end_ms > total_duration_ms:
            end_ms = total_duration_ms
            transition_ms = min(transition_ms, start_ms)
        
        # 转换为采样点
        start_sample = int(start_ms / 1000 * sample_rate)
        end_sample = int(end_ms / 1000 * sample_rate)
        
        # 确保不超过音频长度
        start_sample = min(start_sample, total_samples - 1)
        end_sample = min(end_sample, total_samples)
        
        # 裁剪音频
        if start_sample >= end_sample:
            # 创建空音频，保持3D格式 [1, 1, 1]
            segmented_waveform = torch.zeros(batch_size, num_channels, 1, device=waveform.device)
        else:
            segmented_waveform = waveform[:, :, start_sample:end_sample].clone()
        
        # 应用淡入淡出效果
        if transition_ms > 0 and segmented_waveform.shape[-1] > 0:
            fade_out_samples = int(transition_ms / 1000 * sample_rate)
            fade_out_samples = min(fade_out_samples, segmented_waveform.shape[-1])
            
            fade = Fade(fade_in_len=0, 
                        fade_out_len=fade_out_samples,
                        fade_shape="linear")
            
            # 应用淡出效果到每个batch和通道
            for b in range(batch_size):
                for c in range(num_channels):
                    segmented_waveform[b, c] = fade(segmented_waveform[b, c])
        
        # 确保输出格式与原生节点兼容 (batch, channels, samples)
        return ({"waveform": segmented_waveform, "sample_rate": sample_rate},)

class AudioDurationToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("INT", {"default": 25, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_frames",)
    FUNCTION = "calculate_frames"
    CATEGORY = "audio/utils"

    def calculate_frames(self, audio, frame_rate):
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 确定样本数所在的维度
        if waveform.ndim == 3:
            # 3D: [batch, channels, samples]
            total_samples = waveform.shape[2]
        elif waveform.ndim == 2:
            # 2D: [channels, samples]
            total_samples = waveform.shape[1]
        else:
            # 1D: [samples] 或其他情况
            total_samples = waveform.shape[0]
        
        # 计算音频总时长（毫秒）
        duration_ms = (total_samples / sample_rate) * 1000
        
        # 计算总帧数（四舍五入取整）
        total_frames = round(duration_ms * frame_rate / 1000)
        
        return (int(total_frames),)