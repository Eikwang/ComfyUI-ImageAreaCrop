# 图像传输节点
from .common_imports import torch


class ImageTransferNode:
    def __init__(self):
        self.cached_images = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_manual": ("BOOLEAN", {"default": False, "tooltip": "是否合并手动图像"}),
                "merge_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 9999,
                    "step": 1,
                    "tooltip": "插入位置"
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "输入图像（更新缓存）"}),
                "manual_image": ("IMAGE", {"tooltip": "手动图像"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
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
