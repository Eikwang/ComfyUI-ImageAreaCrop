import torch
import numpy as np
import json
import cv2
from torchvision.transforms.functional import resize as torch_resize

class ImageAreaCropNode:
    """图像区域裁切节点 - 从输入图像中裁剪指定区域"""
    
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
        B, H, W, C = image.shape
        cropped_images = []
        crop_info_list = []
        
        for i in range(B):
            # 自动修正越界的裁切区域
            x_i = max(0, min(x, W - 1))
            y_i = max(0, min(y, H - 1))
            width_i = min(width, W - x_i)
            height_i = min(height, H - y_i)
            
            # 裁切操作 (使用PyTorch原生裁切)
            cropped = image[i, y_i:y_i+height_i, x_i:x_i+width_i, :]
            
            if do_resize:
                # 确保正确的维度顺序
                cropped = cropped.permute(2, 0, 1)  # 将通道维度放到前面 (C, H, W)
                cropped = torch_resize(cropped.unsqueeze(0), [scaled_height, scaled_width])[0]
                cropped = cropped.permute(1, 2, 0)  # 恢复维度顺序 (H, W, C)
                
            cropped_images.append(cropped)
            
            crop_info = {
                "original_size": [W, H],
                "crop_position": [x_i, y_i],
                "crop_size": [width_i, height_i],
                "do_resize": do_resize,
                "scaled_size": [scaled_width, scaled_height] if do_resize else None
            }
            crop_info_list.append(crop_info)
            
        result = torch.stack(cropped_images)
        return (result, crop_info_list,)


class AreaCropRestoreNode:
    """区域裁切恢复节点 - 将裁切后的图像拼接到原图指定位置"""
    
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
        B, H, W, C = target_image.shape
        restored_images = []
        
        # 确保裁切信息是列表格式
        if not isinstance(crop_info, list):
            crop_info = [crop_info]
            
        for i in range(B):
            info = crop_info[i] if i < len(crop_info) else crop_info[0]
            target = target_image[i].clone()
            cropped = cropped_image[i]
            
            # 解析裁切信息
            x, y = info["crop_position"]
            width, height = info["crop_size"]
            resize_enabled = info.get("do_resize", False)
            
            # 自动修正越界的恢复区域
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            width = min(width, W - x)
            height = min(height, H - y)
            
            # 如果需要恢复原始尺寸
            if resize_enabled:
                scaled_w, scaled_h = info["scaled_size"]
                # 确保正确的维度顺序
                cropped = cropped.permute(2, 0, 1)  # 将通道维度放到前面 (C, H, W)
                cropped = torch_resize(cropped.unsqueeze(0), [height, width])[0]
                cropped = cropped.permute(1, 2, 0)  # 恢复维度顺序 (H, W, C)
            
            # 确保裁切图像尺寸正确
            if cropped.shape[0] != height or cropped.shape[1] != width:
                # 再次确保正确的维度顺序
                cropped = cropped.permute(2, 0, 1)  # (C, H, W)
                cropped = torch_resize(cropped.unsqueeze(0), [height, width])[0]
                cropped = cropped.permute(1, 2, 0)  # (H, W, C)
            
            # 直接修改目标图像
            target[y:y+height, x:x+width, :] = cropped
            restored_images.append(target)
        
        return (torch.stack(restored_images),)