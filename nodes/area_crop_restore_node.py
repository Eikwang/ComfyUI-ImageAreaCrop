# 区域裁切恢复节点
from .common_imports import interpolate


class AreaCropRestoreNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_image": ("IMAGE", {"tooltip": "裁切后的图像"}),
                "target_image": ("IMAGE", {"tooltip": "要贴回的目标图像"}),
                "crop_info": ("JSON", {"tooltip": "裁切信息（位置与尺寸）"}),
            },
            "optional": {
                "scale_info": ("JSON", {"tooltip": "缩放信息（如裁切阶段启用缩放）"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore_image"
    RETURN_NAMES = ("复原图像",)
    CATEGORY = "image"

    def restore_image(self, cropped_image, target_image, crop_info, scale_info=None):
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
            # 如果提供了缩放信息且启用缩放，先回缩到目标尺寸
            use_scale = False
            if scale_info is not None:
                si = scale_info[0] if isinstance(scale_info, list) else scale_info
                use_scale = bool(si.get("enabled", False))
            if use_scale and (group_crops.shape[1] != target_h or group_crops.shape[2] != target_w):
                group_crops = group_crops.permute(0, 3, 1, 2)
                group_crops = interpolate(
                    group_crops,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                group_crops = group_crops.permute(0, 2, 3, 1)
            
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
