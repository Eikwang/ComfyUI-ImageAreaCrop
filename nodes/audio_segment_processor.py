# 音频片段处理节点
from .common_imports import torch, os, torchaudio, logger
from typing import List, Dict, Any


class AudioSegmentProcessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_segments": ("AUDIO_LIST", {"tooltip": "音频片段列表"}),
                "segment_info": ("JSON", {"tooltip": "片段信息JSON"}),
                "merge_segments": ("BOOLEAN", {"default": True, "tooltip": "是否合并片段"}),
                "save_segments": ("BOOLEAN", {"default": False, "tooltip": "是否保存片段"}),
                "selected_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "选定片段索引"}),
                "filename_prefix": ("STRING", {"default": "segment", "tooltip": "文件名前缀"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("处理后的音频",)
    FUNCTION = "process_audio_segments"
    CATEGORY = "audio/processing"
    
    def process_audio_segments(self, audio_segments, segment_info, merge_segments=True, 
                             save_segments=False, selected_index=0, filename_prefix="segment"):
        """处理音频片段主方法"""
        # 输入验证
        if not audio_segments or not isinstance(audio_segments, list):
            logger.error("音频片段列表为空或格式不正确")
            return (self._create_empty_audio(),)
        
        if not segment_info or not isinstance(segment_info, list):
            logger.error("片段信息为空或格式不正确")
            return (self._create_empty_audio(),)
        
        try:
            if merge_segments:
                # 合并所有音频片段
                merged_audio = self._merge_audio_segments(audio_segments)
                
                # 保存合并后的音频（可选）
                if save_segments:
                    self._save_audio(merged_audio, f"{filename_prefix}_merged.wav")
                
                return (merged_audio,)
            else:
                # 选择单个片段
                if selected_index >= len(audio_segments):
                    logger.warning(f"选定索引 {selected_index} 超出范围，使用最后一个片段")
                    selected_index = len(audio_segments) - 1
                
                selected_audio = audio_segments[selected_index]
                
                # 保存选定的音频片段（可选）
                if save_segments:
                    self._save_audio(selected_audio, f"{filename_prefix}_{selected_index}.wav")
                
                return (selected_audio,)
        
        except Exception as e:
            logger.error(f"处理音频片段时发生错误: {str(e)}")
            return (self._create_empty_audio(),)
    
    def _merge_audio_segments(self, audio_segments):
        """合并音频片段"""
        try:
            if not audio_segments:
                return self._create_empty_audio()
            
            # 获取第一个片段的采样率
            first_segment = audio_segments[0]
            if "sample_rate" not in first_segment:
                logger.error("音频片段缺少采样率信息")
                return self._create_empty_audio()
            
            sample_rate = first_segment["sample_rate"]
            waveforms = []
            
            for i, segment in enumerate(audio_segments):
                try:
                    if "waveform" not in segment:
                        logger.warning(f"片段 {i} 缺少波形数据，跳过")
                        continue
                    
                    waveform = segment["waveform"]
                    
                    # 确保波形是3D张量 [batch, channels, samples]
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim != 3:
                        logger.warning(f"片段 {i} 波形维度异常: {waveform.ndim}，尝试重塑")
                        waveform = waveform.reshape(1, 1, -1)
                    
                    # 检查采样率一致性
                    if segment.get("sample_rate", sample_rate) != sample_rate:
                        logger.warning(f"片段 {i} 采样率不一致，进行重采样")
                        try:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=segment["sample_rate"], 
                                new_freq=sample_rate
                            )
                            waveform = resampler(waveform)
                        except Exception as e:
                            logger.error(f"重采样片段 {i} 失败: {str(e)}")
                            continue
                    
                    waveforms.append(waveform)
                
                except Exception as e:
                    logger.error(f"处理片段 {i} 时发生错误: {str(e)}")
                    continue
            
            if not waveforms:
                logger.error("没有有效的音频片段可以合并")
                return self._create_empty_audio()
            
            # 合并波形
            merged_waveform = torch.cat(waveforms, dim=2)  # 在时间维度上连接
            
            return {
                "waveform": merged_waveform,
                "sample_rate": sample_rate
            }
        
        except Exception as e:
            logger.error(f"合并音频片段时发生错误: {str(e)}")
            return self._create_empty_audio()
    
    def _save_audio(self, audio, filename):
        """保存音频文件"""
        try:
            if "waveform" not in audio or "sample_rate" not in audio:
                logger.error("音频数据格式不正确，无法保存")
                return
            
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # 确保波形是2D张量 [channels, samples] 用于保存
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)  # 移除batch维度
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # 添加channel维度
            
            # 创建输出目录
            output_dir = "output/audio_segments"
            os.makedirs(output_dir, exist_ok=True)
            
            # 完整文件路径
            filepath = os.path.join(output_dir, filename)
            
            # 保存音频文件
            torchaudio.save(filepath, waveform, sample_rate)
            logger.info(f"音频已保存到: {filepath}")
        
        except Exception as e:
            logger.error(f"保存音频文件时发生错误: {str(e)}")
    
    def _create_empty_audio(self):
        """创建空音频对象"""
        try:
            empty_waveform = torch.zeros(1, 1, 1)
            return {
                "waveform": empty_waveform,
                "sample_rate": 44100
            }
        except Exception as e:
            logger.error(f"创建空音频时发生错误: {str(e)}")
            return {
                "waveform": torch.zeros(1, 1, 1),
                "sample_rate": 44100
            }
