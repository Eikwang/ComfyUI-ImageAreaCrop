# 图像区域裁切节点 - 支持可视化区域选择
from .common_imports import torch, interpolate
from server import PromptServer
from aiohttp import web
from PIL import Image
import io, base64
from threading import Event
crop_node_data = {}


class ImageAreaCropNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
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

    RETURN_TYPES = ("IMAGE", "JSON", "JSON")
    RETURN_NAMES = ("图像", "裁切信息", "缩放信息")
    FUNCTION = "crop_image"
    CATEGORY = "image"

    def crop_image(self, image, do_resize=False, scaled_width=512, scaled_height=512, unique_id=None):
        # 确保在GPU上处理
        device = image.device
        B, H, W, C = image.shape
        need_open = False
        rect = None
        if unique_id is not None:
            key = str(unique_id)
            state = crop_node_data.get(key)
            rect = state.get("crop_rect") if state else None
            processing_complete = state.get("processing_complete") if state else False
            # 需要重新弹出选择界面的条件：状态不存在、裁切矩形不存在、或者处理未完成
            if state is None or rect is None or not processing_complete:
                need_open = True
        if need_open:
            try:
                from server import PromptServer
                if key not in crop_node_data:
                    crop_node_data[key] = {"event": Event(), "result": None, "processing_complete": False}
                else:
                    crop_node_data[key]["event"] = Event()
                    crop_node_data[key]["result"] = None
                    crop_node_data[key]["processing_complete"] = False
                img_np = (torch.clamp(image[0].clone(), 0, 1) * 255).cpu().numpy().astype('uint8')
                pil_image = Image.fromarray(img_np)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                crop_node_data[key]["last_image"] = f"data:image/png;base64,{b64}"
                PromptServer.instance.send_sync("image_area_crop_update", {"node_id": unique_id, "image_data": f"data:image/png;base64,{b64}"})
                crop_node_data[key]["event"].wait(timeout=60)
                rect = crop_node_data.get(key, {}).get("crop_rect")
            except Exception:
                pass

        # 使用已保存的裁切参数或安全默认
        if rect and len(rect) == 4:
            x, y, width, height = rect
        else:
            x, y = 0, 0
            width, height = W, H

        # 边界修正
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        width = max(1, min(int(width), W - x))
        height = max(1, min(int(height), H - y))
        
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

        # 缩放信息（用于复原节点回缩）
        scale_info = {
            "enabled": bool(do_resize),
            "source_size": [width, height],
            "target_size": [scaled_width, scaled_height] if do_resize else [width, height],
            "scale_factor": [
                (scaled_width / width) if (do_resize and width > 0) else 1.0,
                (scaled_height / height) if (do_resize and height > 0) else 1.0,
            ],
        }
        scale_info_list = [scale_info] * B
        
        return (cropped, crop_info_list, scale_info_list)

@PromptServer.instance.routes.post("/image_cropper/apply")
async def apply_image_cropper(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        key = str(node_id)
        x = int(data.get("x", 0) or 0)
        y = int(data.get("y", 0) or 0)
        w = int(data.get("width", 0) or 0)
        h = int(data.get("height", 0) or 0)
        if key not in crop_node_data:
            crop_node_data[key] = {"event": Event(), "result": None, "processing_complete": False}
        crop_node_data[key]["crop_rect"] = [x, y, w, h]
        crop_node_data[key]["processing_complete"] = True
        crop_node_data[key]["event"].set()
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})

@PromptServer.instance.routes.post("/image_cropper/cancel")
async def cancel_crop(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        key = str(node_id)
        if key in crop_node_data:
            if "event" in crop_node_data[key]:
                crop_node_data[key]["event"].set()
            # 取消操作时重置处理完成状态，确保下次执行会重新弹出选择界面
            crop_node_data[key]["processing_complete"] = False
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})

@PromptServer.instance.routes.post("/image_cropper/clear")
async def clear_crop(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        key = str(node_id)
        if key in crop_node_data:
            # 完全重置节点状态，确保下次执行时会重新弹出选择界面
            crop_node_data[key]["crop_rect"] = None
            crop_node_data[key]["processing_complete"] = False
            # 清除事件状态，确保新的等待可以正常进行
            if "event" in crop_node_data[key]:
                crop_node_data[key]["event"].clear()
        else:
            crop_node_data[key] = {"event": Event(), "result": None, "processing_complete": False, "crop_rect": None}
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})
@PromptServer.instance.routes.post("/image_cropper/get")
async def get_crop_state(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        key = str(node_id)
        state = crop_node_data.get(key, {})
        return web.json_response({
            "crop_rect": state.get("crop_rect"),
            "image_data": state.get("last_image")
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})
