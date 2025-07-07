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

import audioop
import tempfile
from comfy import model_management
import torchaudio
from torchaudio.transforms import Fade

from scipy.signal import butter, lfilter
import webrtcvad
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioSpeechSegmenter")
import comfy.utils
from typing import List, Dict, Tuple, Optional, Any
import ffmpeg
import hashlib

from tqdm import tqdm
import comfy.sd
import sys
import comfy.samplers
import nodes

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
        
        # 规范化波形维度为3D (batch, channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
        elif waveform.ndim == 3:
            pass  # 已经是正确的3D格式
        else:
            # 其他维度 - 尝试转换为3D
            waveform = waveform.reshape(1, 1, -1)
        
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
        
        # 确定波形维度
        if waveform.ndim == 3:  # [batch, channels, samples]
            total_samples = waveform.shape[2]
        elif waveform.ndim == 2:  # [channels, samples]
            total_samples = waveform.shape[1]
        elif waveform.ndim == 1:  # [samples]
            total_samples = waveform.shape[0]
        else:
            # 未知维度，使用默认值
            total_samples = waveform.shape[-1] if waveform.ndim > 0 else 0
        
        # 计算音频总时长（毫秒）
        duration_ms = (total_samples / sample_rate) * 1000
        
        # 计算总帧数（四舍五入取整）
        total_frames = round(duration_ms * frame_rate / 1000)
        
        # 确保至少有一帧
        total_frames = max(1, total_frames)
        
        return (int(total_frames),)
    


class AudioSpeechSegmenter:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),                # 输入音频
                "frame_duration": ("INT", {"default": 20, "min": 10, "max": 30, "step": 10}),  # VAD帧时长(ms)
                "min_speech_duration": ("FLOAT", {"default": 1, "min": 0.1, "max": 5.0, "step": 0.1}),  # 最小语音时长(s)
                "max_silence_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),  # 最大静音时长(s)
                "aggressiveness": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),  # VAD攻击性级别
                "max_segment_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 1.0}),  # 最大片段时长(s)
            },
            "optional": {
                "min_energy_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001}),  # 最小能量阈值
                "noise_level_db": ("FLOAT", {"default": -40, "min": -90, "max": -30, "step": 1}),  # 噪音水平阈值(dB)
                "noise_reduction": ("BOOLEAN", {"default": False}),  # 噪音抑制开关声抑制开关
                "adaptive_threshold": ("BOOLEAN", {"default": False}),  # 自适应阈值开关
                "resample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000}),  # 重采样率
                "debug_output": ("BOOLEAN", {"default": False}),  # 调试输出开关
            }
        }
    
    RETURN_TYPES = ("AUDIO_LIST", "JSON", "INT")
    RETURN_NAMES = ("segments", "segment_info", "segment_count")
    FUNCTION = "segment_audio"
    CATEGORY = "audio/processing"
    
    def segment_audio(self, audio, frame_duration, min_speech_duration, max_silence_duration, 
                     aggressiveness, max_segment_duration, min_energy_threshold=0.01, 
                     noise_level_db=-60, noise_reduction=True, adaptive_threshold=True,
                     resample_rate=16000, debug_output=False):
        """音频语音分割主方法"""
        # 输入验证
        if "waveform" not in audio or "sample_rate" not in audio:
            logger.error("输入音频格式不正确")
            return (self.handle_invalid_input(), [], 0)
        
        # 获取音频数据
        waveform = audio["waveform"]
        original_sample_rate = audio["sample_rate"]
        logger.info(f"开始音频分割处理，采样率: {original_sample_rate}Hz")
        
        try:
            # 统一波形维度为3D [batch, channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
                logger.info("输入音频维度: 1D -> 3D (1,1,N)")
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                logger.info(f"输入音频维度: 2D -> 3D (1,{waveform.shape[1]},N)")
            elif waveform.ndim != 3:
                waveform = waveform.reshape(1, 1, -1)
                logger.warning(f"未知音频维度 {waveform.ndim}D，转换为3D (1,1,N)")
            
            batch_size, num_channels, total_samples = waveform.shape
            logger.info(f"处理后音频维度: batch={batch_size}, channels={num_channels}, samples={total_samples}")
            
            # 转换为单声道处理
            if num_channels > 1:
                logger.info(f"将{num_channels}声道音频转换为单声道")
                waveform = waveform.mean(dim=1, keepdim=True)
                num_channels = 1
            
            # 重采样为VAD支持的采样率
            if original_sample_rate != resample_rate:
                logger.info(f"将音频从{original_sample_rate}Hz重采样到{resample_rate}Hz")
                try:
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=resample_rate)
                    vad_waveform = resampler(waveform)
                    resampled_samples = vad_waveform.shape[-1]
                    logger.info(f"重采样后样本数: {resampled_samples} (原始: {total_samples})")
                except Exception as e:
                    logger.error(f"重采样失败: {str(e)}")
                    vad_waveform = waveform.clone()
                    resample_rate = original_sample_rate
            else:
                vad_waveform = waveform.clone()
                resample_rate = original_sample_rate
            
            # 转换为NumPy数组进行处理
            audio_np = vad_waveform.cpu().numpy().squeeze()
            
            # 应用噪声抑制（可选）
            if noise_reduction:
                logger.info("应用高级噪声抑制")
                try:
                    audio_np = self.apply_advanced_noise_reduction(audio_np, resample_rate, noise_level_db)
                except Exception as e:
                    logger.error(f"噪声抑制失败: {str(e)}")
            
            # 估算噪声基线
            noise_floor = self.estimate_noise_floor(audio_np, resample_rate)
            logger.info(f"噪声基底估计: {noise_floor:.6f} ({20*np.log10(noise_floor+1e-10):.1f} dB)")
            
            # 使用WebRTC VAD检测语音活动
            segments = self.enhanced_vad_detection(
                audio_np, resample_rate, frame_duration, aggressiveness, 
                min_speech_duration, max_silence_duration, max_segment_duration,
                min_energy_threshold, noise_floor, adaptive_threshold, debug_output
            )
            
            # 转换为Tensor格式
            segment_tensors = []
            segment_info = []
            
            logger.info(f"检测到 {len(segments)} 个语音片段")
            
            for i, (start, end) in enumerate(segments):
                try:
                    # 计算原始音频中的样本点位置
                    start_sample = min(total_samples - 1, int(start * original_sample_rate))
                    end_sample = min(total_samples, int(end * original_sample_rate))
                    duration_samples = end_sample - start_sample
                    
                    # 确保片段长度有效
                    if end_sample <= start_sample or duration_samples < 10:
                        logger.warning(f"跳过无效片段 #{i+1}: 开始={start_sample}, 结束={end_sample}")
                        continue
                    
                    # 裁剪音频片段
                    seg_waveform = waveform[:, :, start_sample:end_sample].clone()
                    duration_sec = duration_samples / original_sample_rate
                    logger.info(f"片段 #{i+1}: 开始={start:.2f}s, 结束={end:.2f}s, 时长={duration_sec:.2f}s")
                    
                    # 确保输出是3D张量 (batch, channels, samples)
                    if seg_waveform.ndim == 2:
                        seg_waveform = seg_waveform.unsqueeze(0)
                    elif seg_waveform.ndim == 1:
                        seg_waveform = seg_waveform.unsqueeze(0).unsqueeze(0)
                    
                    segment_tensors.append({
                        "waveform": seg_waveform,
                        "sample_rate": original_sample_rate
                    })
                    
                    segment_info.append({
                        "index": i,
                        "start_time": start,
                        "end_time": end,
                        "duration": duration_sec,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "original_sample_rate": original_sample_rate,
                        "original_duration": total_samples / original_sample_rate  # 原始总时长
                    })
                    
                except Exception as e:
                    logger.error(f"处理片段 #{i+1} 时发生错误: {str(e)}")
            
            # 处理没有检测到语音的情况
            if not segment_tensors:
                logger.warning("未检测到任何语音片段!")
                # 创建空的3D张量
                empty_waveform = torch.zeros(1, 1, 1)
                segment_tensors = [{"waveform": empty_waveform, "sample_rate": original_sample_rate}]
                segment_info = [{
                    "index": 0,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "duration": 0.0,
                    "start_sample": 0,
                    "end_sample": 0
                }]
            
            segment_count = len(segment_tensors)
            logger.info(f"处理完成，输出 {segment_count} 个音频片段")
            
            return (segment_tensors, segment_info, segment_count)
        
        except Exception as e:
            logger.error(f"音频分割过程中发生错误: {str(e)}")
            return (self.handle_invalid_input(), [], 0)
    
    def apply_advanced_noise_reduction(self, audio, sample_rate, noise_level_db):
        """应用更高级的噪声抑制算法，特别针对细微噪音"""
        try:
            # 1. 频谱减法
            stft = librosa.stft(audio, n_fft=512, hop_length=128)
            magnitude, phase = np.abs(stft), np.angle(stft)
            
            # 估计噪声谱（使用前10帧作为噪声样本）
            noise_frames = min(10, magnitude.shape[1])
            noise_est = np.median(magnitude[:, :noise_frames], axis=1)
            
            # 应用谱减法
            beta = 2.0  # 过减因子
            denoised_mag = np.maximum(magnitude - beta * noise_est[:, np.newaxis], 0)
            
            # 2. 应用自适应噪声门
            noise_threshold = 10 ** (noise_level_db / 20)  # 将dB转换为幅度值
            denoised_mag[denoised_mag < noise_threshold] = 0
            
            # 重建信号
            denoised_stft = denoised_mag * np.exp(1j * phase)
            denoised_audio = librosa.istft(denoised_stft, hop_length=128)
            
            return denoised_audio
        except Exception as e:
            logger.error(f"高级噪声抑制失败: {str(e)}")
            return audio
    
    def estimate_noise_floor(self, audio, sample_rate, percentile=25):
        """估算噪声基底（能量分布的百分位数）"""
        try:
            # 计算每帧的能量
            frame_size = int(sample_rate * 0.02)  # 20ms帧
            n_frames = len(audio) // frame_size
            energies = np.zeros(n_frames)
            
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                energies[i] = np.sqrt(np.mean(frame**2))
            
            # 取低百分位数作为噪声基底
            return np.percentile(energies, percentile)
        except Exception as e:
            logger.error(f"噪声基底估计失败: {str(e)}")
            return 0.001  # 默认值
    
    def enhanced_vad_detection(self, audio, sample_rate, frame_duration, aggressiveness, 
                             min_speech_duration, max_silence_duration, max_segment_duration,
                             min_energy_threshold, noise_floor, adaptive_threshold, debug_output=False):
        """增强版VAD检测，更好地区分语音和细微噪音"""
        logger.info(f"开始VAD检测: 采样率={sample_rate}Hz, 帧时长={frame_duration}ms")
        logger.info(f"参数: 攻击性={aggressiveness}, 最小语音时长={min_speech_duration}s, 最大静音时长={max_silence_duration}s")
        logger.info(f"自适应阈值: {'启用' if adaptive_threshold else '禁用'}, 噪声基底={noise_floor:.6f}")
        
        try:
            # 检查采样率是否有效
            valid_rates = [8000, 16000, 32000, 48000]
            if sample_rate not in valid_rates:
                # 使用最接近的有效采样率
                closest_rate = min(valid_rates, key=lambda x: abs(x - sample_rate))
                logger.warning(f"采样率{sample_rate}Hz不是VAD支持的采样率，重采样到{closest_rate}Hz")
                try:
                    audio = torchaudio.functional.resample(torch.tensor(audio), sample_rate, closest_rate).numpy()
                    sample_rate = closest_rate
                except:
                    logger.error(f"重采样失败，使用原始采样率 {sample_rate}Hz")
            
            # 确保帧时长是10,20或30ms的整数倍
            valid_frame_durations = [10, 20, 30]
            if frame_duration not in valid_frame_durations:
                closest_duration = min(valid_frame_durations, key=lambda x: abs(x - frame_duration))
                logger.warning(f"帧时长{frame_duration}ms不是VAD支持的时长，使用{closest_duration}ms")
                frame_duration = closest_duration
            
            # 初始化VAD
            vad = webrtcvad.Vad(aggressiveness)
            
            # 计算帧大小
            frame_size = int(sample_rate * frame_duration / 1000)
            audio_length = len(audio)
            total_duration = audio_length / sample_rate
            
            logger.info(f"音频长度: {audio_length}样本 ({total_duration:.2f}秒)")
            
            # 确保音频长度是帧大小的整数倍
            if audio_length % frame_size != 0:
                # 填充音频使其成为整数倍
                pad_length = frame_size - (audio_length % frame_size)
                audio = np.pad(audio, (0, pad_length), 'constant')
                audio_length = len(audio)
                logger.info(f"音频填充: 增加{pad_length}样本, 新长度={audio_length}")
            
            n_frames = audio_length // frame_size
            logger.info(f"总帧数: {n_frames}")
            
            # 检测语音活动
            speech_frames = []
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                
                # 转换为16位PCM (WebRTC VAD要求)
                if frame.dtype != np.int16:
                    # 归一化并转换为16位PCM
                    max_val = np.max(np.abs(frame))
                    if max_val > 0:
                        frame = (frame / max_val) * 32767
                    frame = frame.astype(np.int16)
                
                # VAD基础检测
                try:
                    is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                except Exception as e:
                    logger.error(f"帧#{i+1}处理错误: {str(e)}")
                    is_speech = False
                
                # 计算帧能量
                frame_energy = np.sqrt(np.mean(frame.astype(np.float32)**2))
                
                # 自适应阈值处理
                if adaptive_threshold:
                    # 动态阈值 = 噪声基底 + 6dB (约2倍幅度)
                    dynamic_threshold = max(min_energy_threshold, noise_floor * 2.0)
                    if frame_energy < dynamic_threshold:
                        if is_speech and debug_output:
                            logger.debug(f"帧#{i+1}: 检测到语音但能量过低({frame_energy:.4f}<{dynamic_threshold}), 标记为静音")
                        is_speech = False
                    elif debug_output:
                        logger.debug(f"帧#{i+1}: {'语音' if is_speech else '静音'}, 能量={frame_energy:.4f}")
                else:
                    # 固定阈值处理
                    if frame_energy < min_energy_threshold:
                        if is_speech and debug_output:
                            logger.debug(f"帧#{i+1}: 检测到语音但能量过低({frame_energy:.4f}<{min_energy_threshold}), 标记为静音")
                        is_speech = False
                    elif debug_output:
                        logger.debug(f"帧#{i+1}: {'语音' if is_speech else '静音'}, 能量={frame_energy:.4f}")
                
                # 附加特征：频谱平坦度检测（噪音通常频谱更平坦）
                if is_speech:  # 只对VAD标记为语音的帧进行额外验证
                    try:
                        spectrum = np.abs(np.fft.rfft(frame.astype(np.float32)))
                        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)
                        
                        # 高平坦度表明可能是噪音（语音通常<0.5，噪音>0.7）
                        if spectral_flatness > 0.65:
                            if debug_output:
                                logger.debug(f"帧#{i+1}: VAD标记为语音但频谱平坦({spectral_flatness:.3f})，标记为噪音")
                            is_speech = False
                    except Exception as e:
                        logger.error(f"频谱平坦度计算错误: {str(e)}")
                
                speech_frames.append(is_speech)
            
            # 统计语音帧数量
            speech_count = sum(speech_frames)
            logger.info(f"语音帧: {speech_count}/{n_frames} ({speech_count/n_frames*100:.1f}%)")
            
            # 合并连续语音段
            segments = []
            in_speech = False
            speech_start = 0
            silence_duration = 0
            current_segment_start = 0
            
            for i, is_speech in enumerate(speech_frames):
                time = i * frame_duration / 1000.0  # 当前时间(秒)
                
                if is_speech and not in_speech:
                    # 语音开始
                    in_speech = True
                    speech_start = time
                    silence_duration = 0
                    current_segment_start = time
                    if debug_output:
                        logger.debug(f"帧#{i+1}: 语音开始 @ {time:.3f}s")
                elif not is_speech and in_speech:
                    # 静音中
                    silence_duration += frame_duration / 1000.0
                    
                    # 检查静音是否超过阈值
                    if silence_duration > max_silence_duration:
                        # 结束当前语音段
                        segment_end = time - silence_duration
                        segment_duration = segment_end - current_segment_start
                        
                        if debug_output:
                            logger.debug(f"帧#{i+1}: 静音过长({silence_duration:.3f}s>={max_silence_duration}), 结束语音段")
                        
                        # 检查最小语音时长
                        if segment_duration >= min_speech_duration:
                            # 检查最大段长
                            if segment_duration > max_segment_duration:
                                # 分割过长段落
                                num_subsegments = int(np.ceil(segment_duration / max_segment_duration))
                                subsegment_duration = segment_duration / num_subsegments
                                
                                for j in range(num_subsegments):
                                    start = current_segment_start + j * subsegment_duration
                                    end = start + subsegment_duration
                                    segments.append((start, end))
                                    if debug_output:
                                        logger.debug(f"分割超长段: #{j+1} 开始={start:.3f}, 结束={end:.3f}")
                            else:
                                segments.append((current_segment_start, segment_end))
                                if debug_output:
                                    logger.debug(f"添加语音段: 开始={current_segment_start:.3f}, 结束={segment_end:.3f}, 时长={segment_duration:.3f}s")
                        
                        in_speech = False
                elif not is_speech:
                    silence_duration = 0
            
            # 处理最后一个语音段
            if in_speech:
                end_time = audio_length / sample_rate
                segment_duration = end_time - current_segment_start
                
                if debug_output:
                    logger.debug(f"处理最后一个语音段: 开始={current_segment_start:.3f}, 结束={end_time:.3f}, 时长={segment_duration:.3f}s")
                
                if segment_duration >= min_speech_duration:
                    if segment_duration > max_segment_duration:
                        num_subsegments = int(np.ceil(segment_duration / max_segment_duration))
                        subsegment_duration = segment_duration / num_subsegments
                        
                        for j in range(num_subsegments):
                            start = current_segment_start + j * subsegment_duration
                            end = start + subsegment_duration
                            segments.append((start, end))
                    else:
                        segments.append((current_segment_start, end_time))
            
            # 确保所有片段在音频范围内
            filtered_segments = []
            for start, end in segments:
                if start < 0:
                    start = 0
                if end > total_duration:
                    end = total_duration
                if start < end:
                    filtered_segments.append((start, end))
                else:
                    logger.warning(f"跳过无效片段: 开始({start:.2f}s) >= 结束({end:.2f}s)")
            
            logger.info(f"检测完成，找到 {len(filtered_segments)} 个有效语音段")
            return filtered_segments
        
        except Exception as e:
            logger.error(f"VAD检测过程中发生错误: {str(e)}")
            return []
    
    def handle_invalid_input(self) -> List[Dict]:
        """处理无效输入，返回空音频片段列表"""
        try:
            empty_waveform = torch.zeros(1, 1, 1)
            return [{"waveform": empty_waveform, "sample_rate": 44100}]
        except:
            return [{"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}]



class AudioSegmentProcessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),  # 音频片段列表
                "segment_info": ("JSON",),    # 片段信息
                "merge_audio": ("BOOLEAN", {"default": False}),  # 合并音频开关
                "save_to_local": ("BOOLEAN", {"default": False}),  # 保存到本地开关
                "selected_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),  # 选择的片段索引
                "filename_prefix": ("STRING", {"default": "segments/ComfyUI"}),  # 文件名前缀
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output_audio",)
    FUNCTION = "process_audio_segments"
    CATEGORY = "audio/processing"
    OUTPUT_NODE = True
    
    def process_audio_segments(self, audio_list, segment_info, merge_audio, save_to_local, selected_index, filename_prefix):
        """处理音频片段的主方法"""
        # 输入验证
        if not isinstance(audio_list, list) or len(audio_list) == 0:
            logger.error("输入音频列表为空或格式不正确")
            return (self.handle_invalid_input(audio_list[0] if audio_list else None),)
        
        segment_count = len(audio_list)
        
        try:
            if merge_audio:
                # 合并所有音频片段
                output_audio = self.merge_audio_segments(audio_list)
                # 保存合并后的音频
                if save_to_local:
                    self.save_merged_audio(output_audio, segment_info, filename_prefix)
            else:
                # 修正索引范围
                selected_index = max(0, min(selected_index, segment_count - 1))
                # 获取选中的音频
                output_audio = audio_list[selected_index]
                # 保存所有分段
                if save_to_local:
                    self.save_all_segments(audio_list, segment_info, filename_prefix)
        except Exception as e:
            logger.error(f"处理音频片段时发生错误: {str(e)}")
            return (self.handle_invalid_input(audio_list[0] if audio_list else None),)
        
        return (output_audio,)
    
    def merge_audio_segments(self, audio_list: List[Dict]) -> Dict:
        """合并多个音频片段为一个连续音频"""
        if not audio_list:
            logger.warning("音频列表为空，无法合并")
            return self.handle_invalid_input()
        
        # 获取采样率（假设所有片段采样率相同）
        sample_rate = audio_list[0]["sample_rate"]
        waveforms = []
        first_channels = None
        
        for audio_segment in audio_list:
            try:
                waveform = audio_segment["waveform"]
                # 统一波形维度为3D [batch, channels, samples]
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
                elif waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                elif waveform.ndim == 3:
                    pass  # 已经是正确格式
                else:
                    logger.warning(f"跳过无效维度的音频片段: {waveform.ndim}D")
                    continue
                
                # 记录通道数
                if first_channels is None:
                    first_channels = waveform.shape[1]
                elif waveform.shape[1] != first_channels:
                    logger.warning("通道数不一致，跳过该片段")
                    continue
                
                waveforms.append(waveform)
            except Exception as e:
                logger.error(f"处理音频片段时发生错误: {str(e)}")
        
        if not waveforms:
            logger.warning("没有有效的音频片段可合并")
            return audio_list[0] if audio_list else self.handle_invalid_input()
        
        # 在时间维度上连接所有波形
        try:
            combined_waveform = torch.cat(waveforms, dim=2)
            logger.info(f"合并完成，总时长: {combined_waveform.shape[2] / sample_rate:.2f}s")
            return {"waveform": combined_waveform, "sample_rate": sample_rate}
        except Exception as e:
            logger.error(f"合并音频时发生错误: {str(e)}")
            return audio_list[0] if audio_list else self.handle_invalid_input()
    
    def save_merged_audio(self, audio: Dict, segment_info: List[Dict], filename_prefix: str):
        """保存合并后的音频到本地"""
        if not audio or "waveform" not in audio:
            logger.warning("没有有效的音频数据可保存")
            return
        
        # 准备保存路径
        try:
            output_dir = folder_paths.get_output_directory()
            subfolder = os.path.dirname(filename_prefix)
            full_dir = os.path.join(output_dir, subfolder)
            os.makedirs(full_dir, exist_ok=True)
            
            # 生成文件名
            total_duration = audio["waveform"].shape[2] / audio["sample_rate"]
            segment_count = len(segment_info)
            filename = f"{os.path.basename(filename_prefix)}_merged_{segment_count}segments_{total_duration:.1f}s.wav"
            file_path = os.path.join(full_dir, filename)
            
            # 保存音频
            self.save_audio_file(audio, file_path)
            logger.info(f"合并音频已保存至: {file_path}")
        except Exception as e:
            logger.error(f"保存合并音频时发生错误: {str(e)}")
    
    def save_all_segments(self, audio_list: List[Dict], segment_info: List[Dict], filename_prefix: str):
        """保存所有音频分段到本地"""
        if not audio_list:
            logger.warning("没有音频片段可保存")
            return
        
        # 准备保存路径
        try:
            output_dir = folder_paths.get_output_directory()
            subfolder = os.path.dirname(filename_prefix)
            full_dir = os.path.join(output_dir, subfolder)
            os.makedirs(full_dir, exist_ok=True)
            
            # 保存每个分段
            for idx, audio_segment in enumerate(audio_list):
                try:
                    # 获取片段时间信息
                    start_time = segment_info[idx]["start_time"] if idx < len(segment_info) else 0
                    duration = segment_info[idx]["duration"] if idx < len(segment_info) else 0
                    # 生成文件名
                    filename = f"{os.path.basename(filename_prefix)}_segment-{idx+1:03d}_start-{start_time:.1f}s_dur-{duration:.1f}s.wav"
                    file_path = os.path.join(full_dir, filename)
                    # 保存音频
                    self.save_audio_file(audio_segment, file_path)
                except Exception as e:
                    logger.error(f"保存片段 {idx+1} 时发生错误: {str(e)}")
        except Exception as e:
            logger.error(f"保存音频片段时发生错误: {str(e)}")
    
    def save_audio_file(self, audio: Dict, file_path: str):
        """保存单个音频文件到WAV格式"""
        if not audio or "waveform" not in audio:
            logger.warning("没有有效的音频数据可保存")
            return
        
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # 统一波形维度为2D [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
            elif waveform.ndim == 3:
                if waveform.shape[0] > 1:
                    logger.info(f"多批次音频 ({waveform.shape[0]} batches)，只保存第一个批次")
                    waveform = waveform[0]
                else:
                    waveform = waveform.squeeze(0)  # 移除批次维度 -> [channels, samples]
            
            # 确保是2D张量
            if waveform.ndim != 2:
                logger.warning(f"音频维度异常: {waveform.ndim}D，尝试转换为2D")
                waveform = waveform.reshape(1, -1)
            
            # 保存到CPU
            waveform = waveform.cpu()
            # 使用torchaudio保存
            torchaudio.save(file_path, waveform, sample_rate)
            logger.info(f"音频已保存至: {file_path}")
        except Exception as e:
            logger.error(f"保存音频文件时发生错误: {str(e)}")
    
    def handle_invalid_input(self, reference_audio=None) -> Dict:
        """处理无效输入，返回空音频"""
        try:
            if reference_audio and "sample_rate" in reference_audio:
                sample_rate = reference_audio["sample_rate"]
            else:
                sample_rate = 44100  # 默认采样率
            
            return {"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate}
        except:
            return {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}



class AudioSilenceRestorer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_audio": ("AUDIO",),  # 处理后的音频
                "segment_info": ("JSON",),       # 片段信息
            },
            "optional": {
                "noise_level": ("FLOAT", {"default": -60.0, "min": -90.0, "max": -30.0, "step": 1.0}),  # 静音噪声水平(dB)
                "boundary_buffer_ms": ("INT", {"default": 100, "min": 0, "max": 500, "step": 10}),  # 边界缓冲时间(毫秒)
            }
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("restored_audio", "original_duration")
    FUNCTION = "restore_silence"
    CATEGORY = "audio/processing"
    
    def restore_silence(self, processed_audio, segment_info, noise_level=-60.0, boundary_buffer_ms=100):
        """基于segment_info精确恢复原始音频结构，避免片段粘连"""
        # 输入验证
        if "waveform" not in processed_audio or "sample_rate" not in processed_audio:
            logger.error("输入音频格式不正确")
            return self.handle_invalid_input()
        
        if not segment_info:
            logger.error("片段信息为空")
            return self.handle_invalid_input()
        
        try:
            # 确保segment_info是列表类型
            if not isinstance(segment_info, list):
                segment_info = [segment_info]
            
            # 提取关键元数据
            original_sample_rate = segment_info[0].get("original_sample_rate", 44100)
            original_duration = segment_info[0].get("original_duration", 0.0)
            total_samples = int(original_duration * original_sample_rate)
            
            logger.info(f"原始音频信息: 时长={original_duration:.2f}s, 采样率={original_sample_rate}Hz")
            
            # 获取处理后的音频数据
            waveform = processed_audio["waveform"]
            processed_sample_rate = processed_audio["sample_rate"]
            
            # 统一波形维度为3D [batch, channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 3:
                waveform = waveform.reshape(1, 1, -1)
            
            batch_size, num_channels, processed_samples = waveform.shape
            processed_duration = processed_samples / processed_sample_rate
            logger.info(f"处理后音频信息: 时长={processed_duration:.2f}s, 样本数={processed_samples}, 采样率={processed_sample_rate}Hz")
            
            # 关键修复：如果采样率不同，重采样到原始采样率
            if processed_sample_rate != original_sample_rate:
                logger.warning(f"采样率不匹配！处理后音频:{processed_sample_rate}Hz, 原始音频:{original_sample_rate}Hz")
                logger.info(f"重采样处理后的音频到原始采样率")
                
                # 使用torchaudio进行高质量重采样
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=processed_sample_rate,
                    new_freq=original_sample_rate
                )
                processed_samples = waveform.shape[-1]
                processed_sample_rate = original_sample_rate
                logger.info(f"重采样完成: 新样本数={processed_samples}")
            
            # 创建静音音频容器
            restored_waveform = torch.zeros((batch_size, num_channels, total_samples), 
                                          dtype=waveform.dtype, 
                                          device=waveform.device)
            
            # 添加微小的噪声防止完全静音
            if noise_level > -90:
                noise_amplitude = 10 ** (noise_level / 20)
                restored_waveform += torch.randn_like(restored_waveform) * noise_amplitude
                logger.info(f"添加静音噪声: {noise_level:.1f}dB")
            
            # 创建边界定位器
            boundary_finder = self.BoundaryFinder(waveform, processed_sample_rate)
            
            # 计算边界缓冲的样本数
            boundary_buffer_samples = int(boundary_buffer_ms * processed_sample_rate / 1000)
            logger.info(f"边界缓冲: {boundary_buffer_ms}ms -> {boundary_buffer_samples}样本")
            
            # 按开始时间排序片段
            sorted_segments = sorted(segment_info, key=lambda x: x["start_time"])
            
            # 计算处理后音频的总时长（秒）
            processed_audio_duration = processed_samples / processed_sample_rate
            
            # 计算所有片段在原始音频中的总时长
            total_segment_duration = sum(seg["duration"] for seg in sorted_segments)
            
            # 当前在处理后音频中的位置（样本）
            current_position = 0
            
            # 精确分割并插入处理后的音频
            for i, segment in enumerate(sorted_segments):
                try:
                    # 获取原始位置信息
                    start_time = segment["start_time"]
                    segment_duration = segment["duration"]
                    
                    # 计算该片段在处理后音频中的比例
                    segment_ratio = segment_duration / total_segment_duration
                    target_samples = int(segment_ratio * processed_samples)
                    
                    # 查找最佳边界
                    start_position, end_position = boundary_finder.find_best_boundaries(
                        current_position, 
                        target_samples,
                        buffer_samples=boundary_buffer_samples
                    )
                    
                    # 计算实际切割位置
                    segment_samples = end_position - start_position
                    
                    # 计算在原始音频中的位置（样本）
                    start_sample = int(start_time * original_sample_rate)
                    end_sample = start_sample + segment_samples
                    
                    # 检查插入位置是否超出容器范围
                    if start_sample >= total_samples:
                        logger.warning(f"跳过片段{i}: 开始位置{start_sample}超过总长度{total_samples}")
                        continue
                    
                    if end_sample > total_samples:
                        logger.warning(f"调整片段{i}结束位置: {end_sample} -> {total_samples}")
                        end_sample = total_samples
                        # 调整实际插入的样本数
                        segment_samples = total_samples - start_sample
                        if segment_samples <= 0:
                            logger.warning(f"调整后无有效样本，跳过")
                            continue
                    
                    # 从处理后音频提取片段
                    segment_waveform = waveform[:, :, start_position:start_position+segment_samples]
                    
                    # 更新位置
                    current_position = end_position
                    
                    # 插入原始位置
                    restored_waveform[:, :, start_sample:end_sample] = segment_waveform
                    
                    logger.info(f"插入片段[{i}]: "
                               f"原始位置={start_time:.2f}s-{(start_time+segment_duration):.2f}s, "
                               f"音频位置={start_position}-{end_position}, "
                               f"实际时长={segment_samples/processed_sample_rate:.2f}s")
                    
                except Exception as e:
                    logger.error(f"处理片段{i}时出错: {str(e)}")
                    continue
            
            # 检查是否处理完所有音频
            if current_position < processed_samples:
                logger.warning(f"未使用所有处理后的音频: 剩余{processed_samples - current_position}样本")
            
            output_audio = {
                "waveform": restored_waveform,
                "sample_rate": original_sample_rate
            }
            
            restored_duration = restored_waveform.shape[-1] / original_sample_rate
            logger.info(f"恢复完成: 原始时长={original_duration:.2f}s, 恢复后时长={restored_duration:.2f}s")
            
            return (output_audio, original_duration)
        
        except Exception as e:
            logger.error(f"恢复静音时发生错误: {str(e)}")
            return self.handle_invalid_input()
    
    def handle_invalid_input(self) -> tuple:
        """处理无效输入，返回空音频"""
        return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}, 0.0)
    
    class BoundaryFinder:
        """音频边界定位器，解决片段粘连问题"""
        def __init__(self, waveform, sample_rate):
            self.waveform = waveform
            self.sample_rate = sample_rate
            # 转换为单声道并转为numpy数组
            self.audio = waveform.mean(dim=1).squeeze().cpu().numpy()
            self.length = len(self.audio)
        
        def find_best_boundaries(self, current_position, target_samples, buffer_samples=500):
            """
            查找最佳切割边界
            :param current_position: 当前起始位置（样本）
            :param target_samples: 目标片段长度（样本）
            :param buffer_samples: 搜索缓冲样本数
            :return: (实际起始位置, 实际结束位置)
            """
            # 1. 确定搜索区域
            start_position = current_position
            end_position = current_position + target_samples
            
            # 2. 在搜索区域内查找最佳起始边界
            best_start = self.find_boundary(
                max(0, start_position - buffer_samples),
                min(self.length, start_position + buffer_samples),
                type='start'
            )
            
            # 3. 在搜索区域内查找最佳结束边界
            best_end = self.find_boundary(
                max(0, end_position - buffer_samples),
                min(self.length, end_position + buffer_samples),
                type='end'
            )
            
            # 确保结束位置大于起始位置
            if best_end <= best_start:
                best_end = best_start + target_samples
            
            # 确保不超过音频长度
            best_end = min(best_end, self.length)
            
            return best_start, best_end
        
        def find_boundary(self, start_sample, end_sample, type='start'):
            """
            在指定区域内寻找最佳边界点
            :param start_sample: 搜索起始样本
            :param end_sample: 搜索结束样本
            :param type: 'start' 或 'end'，表示寻找起始边界还是结束边界
            :return: 最佳边界点（样本索引）
            """
            # 提取搜索区域
            region_start = int(max(0, start_sample))
            region_end = int(min(self.length, end_sample))
            
            if region_end <= region_start:
                return region_start
            
            search_region = self.audio[region_start:region_end]
            
            # 计算短时能量
            frame_size = 512  # 约11.6ms @44.1kHz
            step_size = frame_size // 2
            num_frames = max(1, (len(search_region) - frame_size) // step_size)
            
            if num_frames == 0:
                return region_start
            
            energies = np.zeros(num_frames)
            
            for i in range(num_frames):
                start_idx = i * step_size
                frame = search_region[start_idx:start_idx+frame_size]
                energies[i] = np.sqrt(np.mean(frame**2))
            
            # 动态计算能量阈值
            energy_threshold = np.percentile(energies, 25)  # 使用25%分位数作为阈值
            
            # 寻找最佳边界点
            if type == 'start':
                # 寻找第一个低于阈值的点（从前往后）
                for i in range(len(energies)):
                    if energies[i] < energy_threshold:
                        return region_start + i * step_size
                return region_start  # 没找到则返回起始位置
            else:
                # 寻找最后一个低于阈值的点（从后往前）
                for i in range(len(energies)-1, -1, -1):
                    if energies[i] < energy_threshold:
                        return region_start + i * step_size + frame_size
                return region_end  # 没找到则返回结束位置