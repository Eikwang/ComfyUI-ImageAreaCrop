import torch
import numpy as np
import json
import cv2
from torchvision.transforms.functional import resize as torch_resize
from torch.nn.functional import interpolate

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