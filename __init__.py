from .nodes import (
    ImageAreaCropNode, 
    AreaCropRestoreNode, 
    ImageReverseOrderNode, 
    ImageTransferNode, 
    AudioSplitNode, 
    AudioDurationToFrames, 
    AudioSpeechSegmenter,
    AudioSegmentProcessor,
    AudioSilenceRestorer,
    VideoFrameCounter,
)

from .SequenceResampler import SequenceResampler

NODE_CLASS_MAPPINGS = {
    "ImageAreaCropNode": ImageAreaCropNode,
    "AreaCropRestoreNode": AreaCropRestoreNode,
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

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAreaCropNode": "图像区域裁切",
    "AreaCropRestoreNode": "区域裁切恢复",
    "ImageReverseOrderNode": "图像倒序循环",
    "ImageTransferNode": "图像中转传输",
    "AudioSplitNode": "音频分割",
    "AudioDurationToFrames": "音频转祯数",
    "AudioSpeechSegmenter": "音频语音分段",
    "AudioSegmentProcessor": "音频分段中转",
    "AudioSilenceRestorer": "音频静音恢复",
    "VideoFrameCounter": "视频帧数计算",
    "SequenceResampler": "序列重组",
}