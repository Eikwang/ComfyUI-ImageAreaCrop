from .common_imports import interpolate


class MaskAreaRestoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cropped_image": ("IMAGE", {"tooltip": "裁切后的图像序列"}),
                "target_image": ("IMAGE", {"tooltip": "要贴回的目标图像序列"}),
                "crop_info": ("JSON", {"tooltip": "裁切信息（每帧位置与尺寸）"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore_image"
    RETURN_NAMES = ("复原图像",)
    CATEGORY = "image"

    def restore_image(self, cropped_image, target_image, crop_info):
        device = cropped_image.device
        B, H, W, C = target_image.shape
        restored = target_image.clone()
        if not isinstance(crop_info, list):
            crop_info = [crop_info] * B
        x_coords = []
        y_coords = []
        target_heights = []
        target_widths = []
        for i in range(B):
            info = crop_info[i] if i < len(crop_info) else crop_info[0]
            x, y = info["crop_position"]
            w, h = info["crop_size"]
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = min(w, W - x)
            h = min(h, H - y)
            x_coords.append(x)
            y_coords.append(y)
            target_widths.append(w)
            target_heights.append(h)
        resized_crops = []
        size_groups = {}
        for i in range(B):
            key = (target_heights[i], target_widths[i])
            if key not in size_groups:
                size_groups[key] = []
            size_groups[key].append(i)
        for (target_h, target_w), indices in size_groups.items():
            group_crops = cropped_image[indices]
            if group_crops.shape[1] != target_h or group_crops.shape[2] != target_w:
                group_crops = group_crops.permute(0, 3, 1, 2)
                resized_group = interpolate(
                    group_crops,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
                resized_group = resized_group.permute(0, 2, 3, 1)
            else:
                resized_group = group_crops
            for i, idx in enumerate(indices):
                resized_crops.append((idx, resized_group[i]))
        resized_crops.sort(key=lambda x: x[0])
        resized_crops = [crop for _, crop in resized_crops]
        for i in range(B):
            current_crop = resized_crops[i]
            if current_crop.shape[0] != target_heights[i] or current_crop.shape[1] != target_widths[i]:
                current_crop = current_crop.permute(2, 0, 1).unsqueeze(0)
                current_crop = interpolate(
                    current_crop,
                    size=(target_heights[i], target_widths[i]),
                    mode='bilinear',
                    align_corners=False,
                )
                current_crop = current_crop.squeeze(0).permute(1, 2, 0)
            restored[i, y_coords[i]:y_coords[i]+target_heights[i], x_coords[i]:x_coords[i]+target_widths[i], :] = current_crop
        return (restored,)
