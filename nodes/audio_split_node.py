# 音频分割节点
from .common_imports import torch, Fade


class AudioSplitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "输入音频"}),
                "frame_rate": ("INT", {"default": 25, "min": 1, "tooltip": "帧率(帧/秒)"}),
                "split_length": ("INT", {"default": 25, "min": 1, "tooltip": "分段长度(帧)"}),
                "skip_length": ("INT", {"default": 0, "min": 0, "tooltip": "跳过长度(帧)"}),
                "transition_length": ("INT", {"default": 0, "min": 0, "tooltip": "过渡长度(帧)"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("分割音频",)
    FUNCTION = "split_audio"
    CATEGORY = "audio/utils"

    def split_audio(self, audio, frame_rate, split_length, skip_length, transition_length):
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 规范化波形维度为3D (batch, channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
        elif waveform.ndim == 3:
            pass  # 已经是正确的3D格式
        else:
            # 其他维度 - 尝试转换为3D
            waveform = waveform.reshape(1, 1, -1)
        
        batch_size, num_channels, total_samples = waveform.shape
        
        # 计算总音频时长（毫秒）
        total_duration_ms = (total_samples / sample_rate) * 1000
        
        # 计算帧时长（毫秒/帧）
        ms_per_frame = 1000 / frame_rate
        
        # 计算跳过位置和分割区间
        skip_ms = skip_length * ms_per_frame
        transition_ms = transition_length * ms_per_frame
        
        # 实际起始位置（应用过渡偏移）
        start_ms = max(0, skip_ms - transition_ms)
        end_ms = start_ms + (split_length + transition_length) * ms_per_frame
        
        # 处理边界情况
        if skip_ms > total_duration_ms:
            start_ms = max(0, total_duration_ms - split_length * ms_per_frame)
            end_ms = total_duration_ms
            transition_ms = 0  # 末尾取消过渡效果
        elif end_ms > total_duration_ms:
            end_ms = total_duration_ms
            transition_ms = min(transition_ms, start_ms)
        
        # 转换为采样点
        start_sample = int(start_ms / 1000 * sample_rate)
        end_sample = int(end_ms / 1000 * sample_rate)
        
        # 确保不超过音频长度
        start_sample = min(start_sample, total_samples - 1)
        end_sample = min(end_sample, total_samples)
        
        # 裁剪音频
        if start_sample >= end_sample:
            # 创建空音频，保持3D格式 [1, 1, 1]
            segmented_waveform = torch.zeros(batch_size, num_channels, 1, device=waveform.device)
        else:
            segmented_waveform = waveform[:, :, start_sample:end_sample].clone()
        
        # 应用淡入淡出效果
        if transition_ms > 0 and segmented_waveform.shape[-1] > 0:
            fade_out_samples = int(transition_ms / 1000 * sample_rate)
            fade_out_samples = min(fade_out_samples, segmented_waveform.shape[-1])
            
            fade = Fade(fade_in_len=0, 
                        fade_out_len=fade_out_samples,
                        fade_shape="linear")
            
            # 应用淡出效果到每个batch和通道
            for b in range(batch_size):
                for c in range(num_channels):
                    segmented_waveform[b, c] = fade(segmented_waveform[b, c])
        
        # 确保输出格式与原生节点兼容 (batch, channels, samples)
        return ({"waveform": segmented_waveform, "sample_rate": sample_rate},)
