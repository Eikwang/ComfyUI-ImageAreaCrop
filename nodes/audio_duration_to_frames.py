# 音频时长转帧数节点
from .common_imports import torch


class AudioDurationToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "输入音频"}),
                "frame_rate": ("INT", {"default": 25, "min": 1, "tooltip": "帧率(帧/秒)"}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("总帧数",)
    FUNCTION = "calculate_frames"
    CATEGORY = "audio/utils"

    def calculate_frames(self, audio, frame_rate):
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 确定波形维度
        if waveform.ndim == 3:  # [batch, channels, samples]
            total_samples = waveform.shape[2]
        elif waveform.ndim == 2:  # [channels, samples]
            total_samples = waveform.shape[1]
        elif waveform.ndim == 1:  # [samples]
            total_samples = waveform.shape[0]
        else:
            # 未知维度，使用默认值
            total_samples = waveform.shape[-1] if waveform.ndim > 0 else 0
        
        # 计算音频总时长（毫秒）
        duration_ms = (total_samples / sample_rate) * 1000
        
        # 计算总帧数（四舍五入取整）
        total_frames = round(duration_ms * frame_rate / 1000)
        
        # 确保至少有一帧
        total_frames = max(1, total_frames)
        
        return (int(total_frames),)
