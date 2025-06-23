from .nodes import ImageAreaCropNode, AreaCropRestoreNode

NODE_CLASS_MAPPINGS = {
    "ImageAreaCropNode": ImageAreaCropNode,
    "AreaCropRestoreNode": AreaCropRestoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAreaCropNode": "图像区域裁切",
    "AreaCropRestoreNode": "区域裁切恢复"
}
