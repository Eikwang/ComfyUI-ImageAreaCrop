import torch
import numpy as np
import json
import cv2
from torchvision.transforms.functional import resize as torch_resize
from torch.nn.functional import interpolate

class ImageAreaCropNode:
    """图像区域裁切节点 - 优化版 (批量处理+GPU加速)"""
    
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
    """区域裁切恢复节点 - 优化版 (批量处理+GPU加速)"""
    
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
        
        # 为批量处理准备坐标
        x_coords = []
        y_coords = []
        widths = []
        heights = []
        resize_flags = []
        scaled_sizes = []
        
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
            widths.append(w)
            heights.append(h)
            resize_flags.append(info.get("do_resize", False))
            scaled_sizes.append(info.get("scaled_size"))
        
        # 批量处理需要resize的图像
        resize_indices = [i for i, flag in enumerate(resize_flags) if flag]
        if resize_indices:
            # 提取需要resize的图像
            to_resize = cropped_image[resize_indices]
            
            # 获取对应的目标尺寸
            target_sizes = [(heights[i], widths[i]) for i in resize_indices]
            
            # 找出所有不同尺寸的组
            unique_sizes = set(target_sizes)
            
            # 按尺寸分组处理
            for size in unique_sizes:
                # 找到当前尺寸对应的索引
                group_indices = [i for i, s in zip(resize_indices, target_sizes) if s == size]
                
                # 批量resize
                resized_group = to_resize[group_indices]
                resized_group = resized_group.permute(0, 3, 1, 2)
                resized_group = interpolate(
                    resized_group, 
                    size=size, 
                    mode='bilinear', 
                    align_corners=False
                )
                resized_group = resized_group.permute(0, 2, 3, 1)
                
                # 更新需要resize的图像
                to_resize[group_indices] = resized_group
            
            # 更新原始图像
            cropped_image[resize_indices] = to_resize
        
        # 批量替换图像区域
        for i in range(B):
            # 最终尺寸检查
            current_height, current_width = cropped_image[i].shape[:2]
            target_height, target_width = heights[i], widths[i]
            
            if current_height != target_height or current_width != target_width:
                # 使用批量操作进行最终调整
                cropped_i = cropped_image[i:i+1].permute(0, 3, 1, 2)
                cropped_i = interpolate(
                    cropped_i, 
                    size=(target_height, target_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                cropped_image[i] = cropped_i.permute(0, 2, 3, 1)[0]
            
            # 替换目标图像中的区域
            restored[i, y_coords[i]:y_coords[i]+heights[i], 
                    x_coords[i]:x_coords[i]+widths[i], :] = cropped_image[i]
        
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
        
        # 预计算索引 - 避免重复创建张量
        # 创建完整正序和倒序索引
        forward_indices = torch.arange(n, device=device)
        backward_indices = torch.flip(forward_indices, [0])
        
        # 创建去尾索引
        dedup_forward = forward_indices[:-1] if deduplicate and n > 1 else forward_indices
        dedup_backward = backward_indices[:-1] if deduplicate and n > 1 else backward_indices
        
        # 确定序列构建方向
        start_segment = dedup_backward if reverse else dedup_forward
        end_segment = forward_indices if reverse else backward_indices
        
        # 计算中间段数量
        mid_segment_count = loop_count - 1
        
        # 使用张量连接代替列表拼接
        segments = [start_segment]
        
        # 中间段处理 - 使用张量操作替代循环
        if mid_segment_count > 0:
            # 创建中间段张量
            mid_segments = []
            
            # 交替添加正序和倒序索引
            for i in range(mid_segment_count):
                if (i % 2 == 0) ^ reverse:  # 使用异或简化条件
                    mid_segments.append(dedup_backward)
                else:
                    mid_segments.append(dedup_forward)
            
            # 批量合并中间段
            mid_tensor = torch.cat(mid_segments)
            segments.append(mid_tensor)
        
        # 添加结束段
        segments.append(end_segment)
        
        # 合并所有段
        combined_indices = torch.cat(segments)
        
        # 使用索引选择代替数据复制
        processed_images = images.index_select(0, combined_indices)
        
        return (processed_images,)