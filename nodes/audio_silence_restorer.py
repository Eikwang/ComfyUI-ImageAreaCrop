# 音频静音恢复节点
from .common_imports import torch, np, torchaudio, logger
from typing import List, Dict, Tuple


class AudioSilenceRestorer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_audio": ("AUDIO", {"tooltip": "处理后的音频"}),  
                "segment_info": ("JSON", {"tooltip": "片段信息"}),       
            },
            "optional": {
                "noise_level": ("FLOAT", {"default": -60.0, "min": -90.0, "max": -30.0, "step": 1.0, "tooltip": "静音噪声水平(dB)"}),  
                "boundary_buffer_ms": ("INT", {"default": 100, "min": 0, "max": 500, "step": 10, "tooltip": "边界缓冲时间(毫秒)"}),  
            }
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("恢复音频", "原始时长")
    FUNCTION = "restore_silence"
    CATEGORY = "audio/processing"
    
    def restore_silence(self, processed_audio, segment_info, noise_level=-60.0, boundary_buffer_ms=100):
        """基于segment_info精确恢复原始音频结构，避免片段粘连"""
        # 输入验证
        if "waveform" not in processed_audio or "sample_rate" not in processed_audio:
            logger.error("输入音频格式不正确")
            return self.handle_invalid_input()
        
        if not segment_info:
            logger.error("片段信息为空")
            return self.handle_invalid_input()
        
        try:
            # 确保segment_info是列表类型
            if not isinstance(segment_info, list):
                segment_info = [segment_info]
            
            # 提取关键元数据
            original_sample_rate = segment_info[0].get("original_sample_rate", 44100)
            original_duration = segment_info[0].get("original_duration", 0.0)
            total_samples = int(original_duration * original_sample_rate)
            
            logger.info(f"原始音频信息: 时长={original_duration:.2f}s, 采样率={original_sample_rate}Hz")
            
            # 获取处理后的音频数据
            waveform = processed_audio["waveform"]
            processed_sample_rate = processed_audio["sample_rate"]
            
            # 统一波形维度为3D [batch, channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 3:
                waveform = waveform.reshape(1, 1, -1)
            
            batch_size, num_channels, processed_samples = waveform.shape
            processed_duration = processed_samples / processed_sample_rate
            logger.info(f"处理后音频信息: 时长={processed_duration:.2f}s, 样本数={processed_samples}, 采样率={processed_sample_rate}Hz")
            
            # 关键修复：如果采样率不同，重采样到原始采样率
            if processed_sample_rate != original_sample_rate:
                logger.warning(f"采样率不匹配！处理后音频:{processed_sample_rate}Hz, 原始音频:{original_sample_rate}Hz")
                logger.info(f"重采样处理后的音频到原始采样率")
                
                # 使用torchaudio进行高质量重采样
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=processed_sample_rate,
                    new_freq=original_sample_rate
                )
                processed_samples = waveform.shape[-1]
                processed_sample_rate = original_sample_rate
                logger.info(f"重采样完成: 新样本数={processed_samples}")
            
            # 创建静音音频容器
            restored_waveform = torch.zeros((batch_size, num_channels, total_samples), 
                                          dtype=waveform.dtype, 
                                          device=waveform.device)
            
            # 添加微小的噪声防止完全静音
            if noise_level > -90:
                noise_amplitude = 10 ** (noise_level / 20)
                restored_waveform += torch.randn_like(restored_waveform) * noise_amplitude
                logger.info(f"添加静音噪声: {noise_level:.1f}dB")
            
            # 创建边界定位器
            boundary_finder = self.BoundaryFinder(waveform, processed_sample_rate)
            
            # 计算边界缓冲的样本数
            boundary_buffer_samples = int(boundary_buffer_ms * processed_sample_rate / 1000)
            logger.info(f"边界缓冲: {boundary_buffer_ms}ms -> {boundary_buffer_samples}样本")
            
            # 按开始时间排序片段
            sorted_segments = sorted(segment_info, key=lambda x: x["start_time"])
            
            # 计算处理后音频的总时长（秒）
            processed_audio_duration = processed_samples / processed_sample_rate
            
            # 计算所有片段在原始音频中的总时长
            total_segment_duration = sum(seg["duration"] for seg in sorted_segments)
            
            # 当前在处理后音频中的位置（样本）
            current_position = 0
            
            # 精确分割并插入处理后的音频
            for i, segment in enumerate(sorted_segments):
                try:
                    # 获取原始位置信息
                    start_time = segment["start_time"]
                    segment_duration = segment["duration"]
                    
                    # 计算该片段在处理后音频中的比例
                    segment_ratio = segment_duration / total_segment_duration
                    target_samples = int(segment_ratio * processed_samples)
                    
                    # 查找最佳边界
                    start_position, end_position = boundary_finder.find_best_boundaries(
                        current_position, 
                        target_samples,
                        buffer_samples=boundary_buffer_samples
                    )
                    
                    # 计算实际切割位置
                    segment_samples = end_position - start_position
                    
                    # 计算在原始音频中的位置（样本）
                    start_sample = int(start_time * original_sample_rate)
                    end_sample = start_sample + segment_samples
                    
                    # 检查插入位置是否超出容器范围
                    if start_sample >= total_samples:
                        logger.warning(f"跳过片段{i}: 开始位置{start_sample}超过总长度{total_samples}")
                        continue
                    
                    if end_sample > total_samples:
                        logger.warning(f"调整片段{i}结束位置: {end_sample} -> {total_samples}")
                        end_sample = total_samples
                        # 调整实际插入的样本数
                        segment_samples = total_samples - start_sample
                        if segment_samples <= 0:
                            logger.warning(f"调整后无有效样本，跳过")
                            continue
                    
                    # 从处理后音频提取片段
                    segment_waveform = waveform[:, :, start_position:start_position+segment_samples]
                    
                    # 更新位置
                    current_position = end_position
                    
                    # 插入原始位置
                    restored_waveform[:, :, start_sample:end_sample] = segment_waveform
                    
                    logger.info(f"插入片段[{i}]: "
                               f"原始位置={start_time:.2f}s-{(start_time+segment_duration):.2f}s, "
                               f"音频位置={start_position}-{end_position}, "
                               f"实际时长={segment_samples/processed_sample_rate:.2f}s")
                    
                except Exception as e:
                    logger.error(f"处理片段{i}时出错: {str(e)}")
                    continue
            
            # 检查是否处理完所有音频
            if current_position < processed_samples:
                logger.warning(f"未使用所有处理后的音频: 剩余{processed_samples - current_position}样本")
            
            output_audio = {
                "waveform": restored_waveform,
                "sample_rate": original_sample_rate
            }
            
            restored_duration = restored_waveform.shape[-1] / original_sample_rate
            logger.info(f"恢复完成: 原始时长={original_duration:.2f}s, 恢复后时长={restored_duration:.2f}s")
            
            return (output_audio, original_duration)
        
        except Exception as e:
            logger.error(f"恢复静音时发生错误: {str(e)}")
            return self.handle_invalid_input()
    
    def handle_invalid_input(self) -> Tuple:
        """处理无效输入，返回空音频"""
        return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}, 0.0)
    
    class BoundaryFinder:
        """音频边界定位器，解决片段粘连问题"""
        def __init__(self, waveform, sample_rate):
            self.waveform = waveform
            self.sample_rate = sample_rate
            # 转换为单声道并转为numpy数组
            self.audio = waveform.mean(dim=1).squeeze().cpu().numpy()
            self.length = len(self.audio)
        
        def find_best_boundaries(self, current_position, target_samples, buffer_samples=500):
            """
            查找最佳切割边界
            :param current_position: 当前起始位置（样本）
            :param target_samples: 目标片段长度（样本）
            :param buffer_samples: 搜索缓冲样本数
            :return: (实际起始位置, 实际结束位置)
            """
            # 1. 确定搜索区域
            start_position = current_position
            end_position = current_position + target_samples
            
            # 2. 在搜索区域内查找最佳起始边界
            best_start = self.find_boundary(
                max(0, start_position - buffer_samples),
                min(self.length, start_position + buffer_samples),
                type='start'
            )
            
            # 3. 在搜索区域内查找最佳结束边界
            best_end = self.find_boundary(
                max(0, end_position - buffer_samples),
                min(self.length, end_position + buffer_samples),
                type='end'
            )
            
            # 确保结束位置大于起始位置
            if best_end <= best_start:
                best_end = best_start + target_samples
            
            # 确保不超过音频长度
            best_end = min(best_end, self.length)
            
            return best_start, best_end
        
        def find_boundary(self, start_sample, end_sample, type='start'):
            """
            在指定区域内寻找最佳边界点
            :param start_sample: 搜索起始样本
            :param end_sample: 搜索结束样本
            :param type: 'start' 或 'end'，表示寻找起始边界还是结束边界
            :return: 最佳边界点（样本索引）
            """
            # 提取搜索区域
            region_start = int(max(0, start_sample))
            region_end = int(min(self.length, end_sample))
            
            if region_end <= region_start:
                return region_start
            
            search_region = self.audio[region_start:region_end]
            
            # 计算短时能量
            frame_size = 512  # 约11.6ms @44.1kHz
            step_size = frame_size // 2
            num_frames = max(1, (len(search_region) - frame_size) // step_size)
            
            if num_frames == 0:
                return region_start
            
            energies = np.zeros(num_frames)
            
            for i in range(num_frames):
                start_idx = i * step_size
                frame = search_region[start_idx:start_idx+frame_size]
                energies[i] = np.sqrt(np.mean(frame**2))
            
            # 动态计算能量阈值
            energy_threshold = np.percentile(energies, 25)  # 使用25%分位数作为阈值
            
            # 寻找最佳边界点
            if type == 'start':
                # 寻找第一个低于阈值的点（从前往后）
                for i in range(len(energies)):
                    if energies[i] < energy_threshold:
                        return region_start + i * step_size
                return region_start  # 没找到则返回起始位置
            else:
                # 寻找最后一个低于阈值的点（从后往前）
                for i in range(len(energies)-1, -1, -1):
                    if energies[i] < energy_threshold:
                        return region_start + i * step_size + frame_size
                return region_end  # 没找到则返回结束位置
