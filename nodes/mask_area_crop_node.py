from .common_imports import torch, interpolate, cv2, np


class MaskAreaCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像序列"}),
                "mask": ("MASK", {"tooltip": "对应帧的遮罩序列"}),
            },
            "optional": {
                "do_resize": ("BOOLEAN", {"default": True, "tooltip": "是否对裁切结果进行缩放"}),
                "scaled_width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "缩放后的宽度"}),
                "scaled_height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "缩放后的高度"}),
                "edge_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "边缘冗余像素(保持3:4比例)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("图像", "裁切信息")
    FUNCTION = "crop_image_by_mask"
    CATEGORY = "image"

    def crop_image_by_mask(self, image, mask, do_resize=True, scaled_width=512, scaled_height=512, edge_padding=0):
        device = image.device
        B, H, W, C = image.shape

        if mask is None:
            mask = torch.zeros((B, H, W), device=device)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            else:
                mask = mask[:, 0]

        r_w = 3
        r_h = 4
        ratio = r_w / r_h

        centers = []
        sizes = []

        for i in range(B):
            m = mask[i] if i < mask.shape[0] else mask[-1]
            m_np = (m.detach().cpu().numpy() > 0.5).astype(np.uint8)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m_np, connectivity=8)

            if num_labels <= 1:
                cx, cy = W // 2, H // 2
                bw, bh = 1, 1
            else:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_idx_rel = int(np.argmax(areas))
                max_idx = max_idx_rel + 1
                x0 = int(stats[max_idx, cv2.CC_STAT_LEFT])
                y0 = int(stats[max_idx, cv2.CC_STAT_TOP])
                bw = int(stats[max_idx, cv2.CC_STAT_WIDTH])
                bh = int(stats[max_idx, cv2.CC_STAT_HEIGHT])
                cx = int(round(centroids[max_idx][0]))
                cy = int(round(centroids[max_idx][1]))

            bw = max(1, bw + 2 * edge_padding)
            bh = max(1, bh + 2 * edge_padding)

            w_i = bw
            h_i = bh
            if w_i / h_i < ratio:
                w_i = int(np.ceil(ratio * h_i))
            elif w_i / h_i > ratio:
                h_i = int(np.ceil(w_i / ratio))

            w_i = min(w_i, W)
            h_i = min(h_i, H)

            centers.append((cx, cy))
            sizes.append((w_i, h_i))

        if do_resize:
            target_sizes = sizes
        else:
            max_w = max(s[0] for s in sizes)
            max_h = max(s[1] for s in sizes)
            if max_w / max_h < ratio:
                max_w = int(np.ceil(ratio * max_h))
            elif max_w / max_h > ratio:
                max_h = int(np.ceil(max_w / ratio))
            max_w = min(max_w, W)
            max_h = min(max_h, H)
            target_sizes = [(max_w, max_h)] * B

        cropped_list = []
        crop_positions = []
        for i in range(B):
            cx, cy = centers[i]
            tw, th = target_sizes[i]
            x_min = int(max(0, min(cx - tw // 2, W - tw)))
            y_min = int(max(0, min(cy - th // 2, H - th)))
            crop = image[i:i+1, y_min:y_min+th, x_min:x_min+tw, :]
            if do_resize:
                crop = crop.permute(0, 3, 1, 2)
                crop = interpolate(
                    crop,
                    size=(scaled_height, scaled_width),
                    mode='bilinear',
                    align_corners=False,
                )
                crop = crop.permute(0, 2, 3, 1)
            cropped_list.append(crop)
            crop_positions.append((x_min, y_min, tw, th))

        result = torch.cat(cropped_list, dim=0)

        crop_info_list = []
        for i in range(B):
            x_min, y_min, tw, th = crop_positions[i]
            crop_info_list.append({
                "original_size": [W, H],
                "crop_position": [x_min, y_min],
                "crop_size": [tw, th],
                "do_resize": do_resize,
                "scaled_size": [scaled_width, scaled_height] if do_resize else None,
            })

        return (result, crop_info_list)
