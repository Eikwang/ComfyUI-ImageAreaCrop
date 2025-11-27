# 图像区域裁切节点 - 支持可视化区域选择
from .common_imports import torch, interpolate


class ImageAreaCropNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 1,
                    "display": "visual_crop",
                    "tooltip": "裁切区域左上角X坐标"
                }),
                "y": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 1,
                    "display": "visual_crop",
                    "tooltip": "裁切区域左上角Y坐标"
                }),
                "width": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 4096, 
                    "step": 1,
                    "display": "visual_crop",
                    "tooltip": "裁切区域宽度"
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 4096, 
                    "step": 1,
                    "display": "visual_crop",
                    "tooltip": "裁切区域高度"
                }),
            },
            "optional": {
                "do_resize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否对裁切后的图像进行缩放"
                }),
                "scaled_width": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 4096, 
                    "step": 1,
                    "tooltip": "缩放后的宽度"
                }),
                "scaled_height": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 4096, 
                    "step": 1,
                    "tooltip": "缩放后的高度"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON",)
    RETURN_NAMES = ("图像", "裁切信息",)
    FUNCTION = "crop_image"
    CATEGORY = "image"

    def crop_image(self, image, x, y, width, height, do_resize=False, scaled_width=512, scaled_height=512):
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
