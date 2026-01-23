# 区域裁切恢复节点
from .common_imports import interpolate, torch


class AreaCropRestoreNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cropped_image": ("IMAGE", {"tooltip": "裁切后的图像"}),
                "target_image": ("IMAGE", {"tooltip": "要贴回的目标图像"}),
                "crop_info": ("JSON", {"tooltip": "裁切信息（位置与尺寸）"}),
                "expand_mode": (["pingpong", "repeat"], {"default": "pingpong", "tooltip": "当处理后图像数量多于原图时的扩展方式"}),
            },
            "optional": {
                "scale_info": ("JSON", {"tooltip": "缩放信息（如裁切阶段启用缩放）"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore_image"
    RETURN_NAMES = ("复原图像",)
    CATEGORY = "image"

    def restore_image(self, cropped_image, target_image, crop_info, expand_mode="pingpong", scale_info=None):
        device = cropped_image.device
        B_target, H, W, C = target_image.shape
        B_cropped = cropped_image.shape[0]
        
        # 根据裁切图像和目标图像的数量差异处理
        if B_cropped < B_target:
            # 情况1: 处理后的图像数量少于目标图像数量，只恢复前B_cropped帧
            restored = target_image.clone()
            effective_batch_size = B_cropped
        elif B_cropped > B_target:
            # 情况2: 处理后的图像数量多于目标图像数量，扩展目标图像
            if expand_mode == "repeat":
                # 重复循环模式: 0,1,2,3,4,0,1,2,3,4,0,1,2,3,4...
                additional_frames = B_cropped - B_target
                cycle_length = B_target
                extended_frames = []
                
                for i in range(additional_frames):
                    idx = i % cycle_length  # 循环使用原始帧索引
                    extended_frames.append(target_image[idx:idx+1])
                
                if extended_frames:
                    extended_part = torch.cat(extended_frames, dim=0)
                    expanded_target = torch.cat([target_image, extended_part], dim=0)
                else:
                    expanded_target = target_image
            else:  # pingpong
                # 真正的往复循环模式: 0,1,2,3,4,3,2,1,0,1,2,3,4...
                additional_frames = B_cropped - B_target
                extended_frames = []
                
                # 构建pingpong序列的生成器，从最大索引开始往复
                def generate_pingpong_sequence():
                    if B_target == 1:  # 只有1帧的特殊情况
                        while True:
                            yield 0
                    else:
                        idx = B_target - 1  # 从最后一帧开始
                        direction = -1  # 从后向前
                        while True:
                            yield idx
                            next_idx = idx + direction
                            
                            # 检查是否到达边界，需要改变方向
                            if next_idx < 0:  # 到达最前，需要转向
                                direction = 1
                                next_idx = 1  # 转向后从第1帧开始
                            elif next_idx >= B_target:  # 到达最后，需要转向
                                direction = -1
                                next_idx = B_target - 2  # 转向后从倒数第2帧开始
                            
                            idx = next_idx
                
                # 生成额外帧的索引
                pingpong_gen = generate_pingpong_sequence()
                for i in range(additional_frames):
                    idx = next(pingpong_gen)
                    extended_frames.append(target_image[idx:idx+1])
                
                if extended_frames:
                    extended_part = torch.cat(extended_frames, dim=0)
                    expanded_target = torch.cat([target_image, extended_part], dim=0)
                else:
                    expanded_target = target_image
            
            restored = expanded_target.clone()
            effective_batch_size = B_cropped
        else:
            # 数量相等，直接使用原目标图像
            restored = target_image.clone()
            effective_batch_size = B_target
        
        # 确保裁切信息是列表格式，并根据实际处理的帧数调整
        if not isinstance(crop_info, list):
            crop_info = [crop_info] * effective_batch_size
        elif len(crop_info) < effective_batch_size:
            # 如果crop_info数量不足，循环使用
            extended_crop_info = []
            for i in range(effective_batch_size):
                extended_crop_info.append(crop_info[i % len(crop_info)])
            crop_info = extended_crop_info
        
        # 为批量处理准备坐标和尺寸
        x_coords = []
        y_coords = []
        target_heights = []
        target_widths = []
        
        for i in range(effective_batch_size):
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
        
        # 使用实际处理的帧数来处理裁剪图像
        processed_cropped = cropped_image[:effective_batch_size]
        
        # 创建一个列表来存储调整后的裁剪图像
        resized_crops = []
        
        # 分组处理相同尺寸的图像以提高效率
        size_groups = {}
        for i in range(effective_batch_size):
            size_key = (target_heights[i], target_widths[i])
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append(i)
        
        # 按尺寸组处理
        for (target_h, target_w), indices in size_groups.items():
            # 获取该组的所有裁剪图像
            group_crops = processed_cropped[indices]
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
        for i in range(effective_batch_size):
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
