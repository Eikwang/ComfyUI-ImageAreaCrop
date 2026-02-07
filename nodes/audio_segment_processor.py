# 音频片段处理节点
from .common_imports import torch, os, torchaudio, logger
from typing import List, Dict, Any


class AudioSegmentProcessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_segments": ("AUDIO_LIST", {"tooltip": "音频片段列表", "label": "音频片段"}),
                "segment_info": ("JSON", {"tooltip": "片段信息JSON", "label": "片段信息"}),
                "merge_segments": ("BOOLEAN", {"default": True, "tooltip": "是否合并片段", "label": "合并片段"}),
                "save_segments": ("BOOLEAN", {"default": False, "tooltip": "是否保存片段", "label": "保存片段"}),
                "selected_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "选定片段索引", "label": "选定索引"}),
                "filename_prefix": ("STRING", {"default": "segment", "tooltip": "文件名前缀", "label": "文件名前缀"}),
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
            # 当save_segments开启时，执行保存操作
            if save_segments:
                if merge_segments:
                    # 合并所有音频片段并保存
                    merged_audio = self._merge_audio_segments(audio_segments)
                    self._save_audio(merged_audio, f"{filename_prefix}_merged")
                    
                    # 输出合并后的完整音频
                    return (merged_audio,)
                else:
                    # 保存所有单独的音频片段
                    for i, audio_segment in enumerate(audio_segments):
                        self._save_audio(audio_segment, f"{filename_prefix}", segment_index=i)
                    
                    # 返回第一个片段作为输出
                    if audio_segments:
                        return (audio_segments[0],)
                    else:
                        return (self._create_empty_audio(),)
            else:
                # 当save_segments关闭时，根据merge_segments决定输出内容
                if merge_segments:
                    # 合并所有音频片段并输出
                    merged_audio = self._merge_audio_segments(audio_segments)
                    return (merged_audio,)
                else:
                    # 根据selected_index返回指定片段
                    if selected_index >= len(audio_segments):
                        logger.warning(f"选定索引 {selected_index} 超出范围，使用最后一个片段")
                        selected_index = len(audio_segments) - 1
                    
                    selected_audio = audio_segments[selected_index]
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
    
    def _save_audio(self, audio, filename, segment_index=None):
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
            
            # 确保波形是CPU张量
            waveform = waveform.cpu()
            
            # 使用ComfyUI的标准输出目录
            from .common_imports import folder_paths
            output_dir = folder_paths.get_output_directory()
            audio_output_dir = os.path.join(output_dir, "audio_segments")
            os.makedirs(audio_output_dir, exist_ok=True)
            
            # 生成唯一文件名（添加时间戳和可选的片段索引）
            import time
            import uuid
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            
            if segment_index is not None:
                unique_filename = f"{filename}_{timestamp}_{unique_id}_{segment_index}.wav"
            else:
                unique_filename = f"{filename}_{timestamp}_{unique_id}.wav"
            
            # 完整文件路径
            filepath = os.path.join(audio_output_dir, unique_filename)
            
            # 保存音频文件
            torchaudio.save(filepath, waveform, sample_rate)
            logger.info(f"音频已保存到: {filepath}")
        
        except Exception as e:
            logger.error(f"保存音频文件时发生错误: {str(e)}")
            # 尝试使用不同的音频格式
            try:
                if "waveform" in audio and "sample_rate" in audio:
                    waveform = audio["waveform"].cpu()
                    sample_rate = audio["sample_rate"]
                    
                    # 如果原始保存失败，尝试调整数值范围
                    if waveform.max() > 1.0 or waveform.min() < -1.0:
                        # 归一化到 [-1, 1] 范围
                        waveform = torch.clamp(waveform, -1.0, 1.0)
                    
                    # 使用ComfyUI的标准输出目录
                    from .common_imports import folder_paths
                    output_dir = folder_paths.get_output_directory()
                    audio_output_dir = os.path.join(output_dir, "audio_segments")
                    os.makedirs(audio_output_dir, exist_ok=True)
                    
                    # 生成唯一文件名
                    import time
                    import uuid
                    timestamp = int(time.time())
                    unique_id = uuid.uuid4().hex[:8]
                    
                    if segment_index is not None:
                        unique_filename = f"{filename}_{timestamp}_{unique_id}_{segment_index}.wav"
                    else:
                        unique_filename = f"{filename}_{timestamp}_{unique_id}.wav"
                    
                    filepath = os.path.join(audio_output_dir, unique_filename)
                    
                    torchaudio.save(filepath, waveform, sample_rate)
                    logger.info(f"音频已使用归一化方式保存到: {filepath}")
            except Exception as e2:
                logger.error(f"音频保存失败: {str(e2)}")
    
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
