# ComfyUI-ImageAreaCrop Plugin
# 图像区域裁剪和音频处理插件

from .nodes import (
    ImageAreaCropNode, AreaCropRestoreNode, ImageReverseOrderNode, 
    ImageTransferNode, AudioSplitNode, AudioDurationToFrames,
    AudioSpeechSegmenter, AudioSegmentProcessor, AudioSilenceRestorer,
    VideoFrameCounter, SequenceResampler,
    MaskAreaCropNode, MaskAreaRestoreNode
)

WEB_DIRECTORY = "web"

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "ImageAreaCropNode": ImageAreaCropNode,
    "AreaCropRestoreNode": AreaCropRestoreNode,
    "MaskAreaCropNode": MaskAreaCropNode,
    "MaskAreaRestoreNode": MaskAreaRestoreNode,
    "ImageReverseOrderNode": ImageReverseOrderNode,
    "ImageTransferNode": ImageTransferNode,
    "AudioSplitNode": AudioSplitNode,
    "AudioDurationToFrames": AudioDurationToFrames,
    "AudioSpeechSegmenter": AudioSpeechSegmenter,
    "AudioSegmentProcessor": AudioSegmentProcessor,
    "AudioSilenceRestorer": AudioSilenceRestorer,
    "VideoFrameCounter": VideoFrameCounter,
    "SequenceResampler": SequenceResampler,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAreaCropNode": "图像区域裁切",
    "AreaCropRestoreNode": "区域裁切复原",
    "MaskAreaCropNode": "遮罩区域裁切",
    "MaskAreaRestoreNode": "遮罩区域复原",
    "ImageReverseOrderNode": "图像倒序循环",
    "ImageTransferNode": "图像传输",
    "AudioSplitNode": "音频切分",
    "AudioDurationToFrames": "音频时长转帧数",
    "AudioSpeechSegmenter": "语音增强分割",
    "AudioSegmentProcessor": "音频片段处理",
    "AudioSilenceRestorer": "音频静音恢复",
    "VideoFrameCounter": "视频帧数计算",
    "SequenceResampler": "序列重采样",
}
