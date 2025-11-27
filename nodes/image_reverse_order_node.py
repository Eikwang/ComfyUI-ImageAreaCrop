# 图像倒序循环节点
from .common_imports import torch


class ImageReverseOrderNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "输入图像序列"}),
                "reverse": ("BOOLEAN", {"default": True, "tooltip": "是否倒序"}),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "是否循环"}),
                "loop_count": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 100,
                    "step": 1,
                    "tooltip": "循环次数"
                }),
                "deduplicate": ("BOOLEAN", {"default": False, "tooltip": "去除尾帧重复"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
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
