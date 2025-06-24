from .nodes import ImageAreaCropNode, AreaCropRestoreNode, ImageReverseOrderNode

NODE_CLASS_MAPPINGS = {
    "ImageAreaCropNode": ImageAreaCropNode,
    "AreaCropRestoreNode": AreaCropRestoreNode,
    "ImageReverseOrderNode": ImageReverseOrderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAreaCropNode": "图像区域裁切",
    "AreaCropRestoreNode": "区域裁切恢复",
    "ImageReverseOrderNode": "图像倒序"
}