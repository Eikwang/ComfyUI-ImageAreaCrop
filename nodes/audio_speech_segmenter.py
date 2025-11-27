# 音频语音分割节点
from .common_imports import torch, np, librosa, webrtcvad, torchaudio, logger
from typing import List, Dict


class AudioSpeechSegmenter:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "输入音频"}),
                "frame_duration": ("INT", {"default": 20, "min": 10, "max": 30, "step": 10, "tooltip": "VAD帧时长(毫秒)"}),
                "min_speech_duration": ("FLOAT", {"default": 1, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "最小语音时长(秒)"}),
                "max_silence_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "最大静音时长(秒)"}),
                "aggressiveness": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1, "tooltip": "VAD灵敏度"}),
                "max_segment_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": "最大片段时长(秒)"}),
            },
            "optional": {
                "min_energy_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001, "tooltip": "最小能量阈值"}),
                "noise_level_db": ("FLOAT", {"default": -40, "min": -90, "max": -30, "step": 1, "tooltip": "噪声水平(dB)"}),
                "noise_reduction": ("BOOLEAN", {"default": False, "tooltip": "噪声抑制"}),
                "adaptive_threshold": ("BOOLEAN", {"default": False, "tooltip": "自适应阈值"}),
                "resample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000, "tooltip": "重采样率(Hz)"}),
                "debug_output": ("BOOLEAN", {"default": False, "tooltip": "调试输出"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO_LIST", "JSON", "INT")
    RETURN_NAMES = ("片段列表", "片段信息", "片段数量")
    FUNCTION = "segment_audio"
    CATEGORY = "audio/processing"
    
    def segment_audio(self, audio, frame_duration, min_speech_duration, max_silence_duration, 
                     aggressiveness, max_segment_duration, min_energy_threshold=0.01, 
                     noise_level_db=-60, noise_reduction=True, adaptive_threshold=True,
                     resample_rate=16000, debug_output=False):
        """音频语音分割主方法"""
        # 输入验证
        if "waveform" not in audio or "sample_rate" not in audio:
            logger.error("输入音频格式不正确")
            return (self.handle_invalid_input(), [], 0)
        
        # 获取音频数据
        waveform = audio["waveform"]
        original_sample_rate = audio["sample_rate"]
        logger.info(f"开始音频分割处理，采样率: {original_sample_rate}Hz")
        
        try:
            # 统一波形维度为3D [batch, channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 3:
                waveform = waveform.reshape(1, 1, -1)
            
            batch_size, num_channels, total_samples = waveform.shape
            
            # 转换为单声道处理
            if num_channels > 1:
                waveform = waveform.mean(dim=1, keepdim=True)
                num_channels = 1
            
            # 重采样为VAD支持的采样率
            if original_sample_rate != resample_rate:
                try:
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=resample_rate)
                    vad_waveform = resampler(waveform)
                except Exception as e:
                    logger.error(f"重采样失败: {str(e)}")
                    vad_waveform = waveform.clone()
                    resample_rate = original_sample_rate
            else:
                vad_waveform = waveform.clone()
            
            # 转换为NumPy数组进行处理
            audio_np = vad_waveform.cpu().numpy().squeeze()
            
            # 应用噪声抑制（可选）
            if noise_reduction:
                try:
                    audio_np = self.apply_advanced_noise_reduction(audio_np, resample_rate, noise_level_db)
                except Exception as e:
                    logger.error(f"噪声抑制失败: {str(e)}")
            
            # 估算噪声基线
            noise_floor = self.estimate_noise_floor(audio_np, resample_rate)
            
            # 使用WebRTC VAD检测语音活动
            segments = self.enhanced_vad_detection(
                audio_np, resample_rate, frame_duration, aggressiveness, 
                min_speech_duration, max_silence_duration, max_segment_duration,
                min_energy_threshold, noise_floor, adaptive_threshold, debug_output
            )
            
            # 转换为Tensor格式
            segment_tensors = []
            segment_info = []
            
            for i, (start, end) in enumerate(segments):
                try:
                    # 计算原始音频中的样本点位置
                    start_sample = min(total_samples - 1, int(start * original_sample_rate))
                    end_sample = min(total_samples, int(end * original_sample_rate))
                    duration_samples = end_sample - start_sample
                    
                    # 确保片段长度有效
                    if end_sample <= start_sample or duration_samples < 10:
                        continue
                    
                    # 裁剪音频片段
                    seg_waveform = waveform[:, :, start_sample:end_sample].clone()
                    duration_sec = duration_samples / original_sample_rate
                    
                    # 确保输出是3D张量
                    if seg_waveform.ndim == 2:
                        seg_waveform = seg_waveform.unsqueeze(0)
                    elif seg_waveform.ndim == 1:
                        seg_waveform = seg_waveform.unsqueeze(0).unsqueeze(0)
                    
                    segment_tensors.append({
                        "waveform": seg_waveform,
                        "sample_rate": original_sample_rate
                    })
                    
                    segment_info.append({
                        "index": i,
                        "start_time": start,
                        "end_time": end,
                        "duration": duration_sec,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "original_sample_rate": original_sample_rate,
                        "original_duration": total_samples / original_sample_rate
                    })
                    
                except Exception as e:
                    logger.error(f"处理片段 #{i+1} 时发生错误: {str(e)}")
            
            # 处理没有检测到语音的情况
            if not segment_tensors:
                empty_waveform = torch.zeros(1, 1, 1)
                segment_tensors = [{"waveform": empty_waveform, "sample_rate": original_sample_rate}]
                segment_info = [{
                    "index": 0,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "duration": 0.0,
                    "start_sample": 0,
                    "end_sample": 0
                }]
            
            segment_count = len(segment_tensors)
            return (segment_tensors, segment_info, segment_count)
        
        except Exception as e:
            logger.error(f"音频分割过程中发生错误: {str(e)}")
            return (self.handle_invalid_input(), [], 0)
    
    def apply_advanced_noise_reduction(self, audio, sample_rate, noise_level_db):
        """应用高级噪声抑制算法"""
        try:
            # 频谱减法
            stft = librosa.stft(audio, n_fft=512, hop_length=128)
            magnitude, phase = np.abs(stft), np.angle(stft)
            
            # 估计噪声谱
            noise_frames = min(10, magnitude.shape[1])
            noise_est = np.median(magnitude[:, :noise_frames], axis=1)
            
            # 应用谱减法
            beta = 2.0
            denoised_mag = np.maximum(magnitude - beta * noise_est[:, np.newaxis], 0)
            
            # 应用自适应噪声门
            noise_threshold = 10 ** (noise_level_db / 20)
            denoised_mag[denoised_mag < noise_threshold] = 0
            
            # 重建信号
            denoised_stft = denoised_mag * np.exp(1j * phase)
            denoised_audio = librosa.istft(denoised_stft, hop_length=128)
            
            return denoised_audio
        except Exception as e:
            logger.error(f"高级噪声抑制失败: {str(e)}")
            return audio
    
    def estimate_noise_floor(self, audio, sample_rate, percentile=25):
        """估算噪声基底"""
        try:
            frame_size = int(sample_rate * 0.02)  # 20ms帧
            n_frames = len(audio) // frame_size
            energies = np.zeros(n_frames)
            
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                energies[i] = np.sqrt(np.mean(frame**2))
            
            return np.percentile(energies, percentile)
        except Exception as e:
            logger.error(f"噪声基底估计失败: {str(e)}")
            return 0.001
    
    def enhanced_vad_detection(self, audio, sample_rate, frame_duration, aggressiveness, 
                             min_speech_duration, max_silence_duration, max_segment_duration,
                             min_energy_threshold, noise_floor, adaptive_threshold, debug_output=False):
        """增强版VAD检测"""
        try:
            # 检查采样率是否有效
            valid_rates = [8000, 16000, 32000, 48000]
            if sample_rate not in valid_rates:
                closest_rate = min(valid_rates, key=lambda x: abs(x - sample_rate))
                try:
                    audio = torchaudio.functional.resample(torch.tensor(audio), sample_rate, closest_rate).numpy()
                    sample_rate = closest_rate
                except:
                    pass
            
            # 确保帧时长有效
            valid_frame_durations = [10, 20, 30]
            if frame_duration not in valid_frame_durations:
                frame_duration = min(valid_frame_durations, key=lambda x: abs(x - frame_duration))
            
            # 初始化VAD
            vad = webrtcvad.Vad(aggressiveness)
            
            # 计算帧大小
            frame_size = int(sample_rate * frame_duration / 1000)
            audio_length = len(audio)
            
            # 确保音频长度是帧大小的整数倍
            if audio_length % frame_size != 0:
                pad_length = frame_size - (audio_length % frame_size)
                audio = np.pad(audio, (0, pad_length), 'constant')
                audio_length = len(audio)
            
            n_frames = audio_length // frame_size
            
            # 检测语音活动
            speech_frames = []
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                
                # 转换为16位PCM
                if frame.dtype != np.int16:
                    max_val = np.max(np.abs(frame))
                    if max_val > 0:
                        frame = (frame / max_val) * 32767
                    frame = frame.astype(np.int16)
                
                # VAD基础检测
                try:
                    is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                except Exception:
                    is_speech = False
                
                # 计算帧能量
                frame_energy = np.sqrt(np.mean(frame.astype(np.float32)**2))
                
                # 自适应阈值处理
                if adaptive_threshold:
                    dynamic_threshold = max(min_energy_threshold, noise_floor * 2.0)
                    if frame_energy < dynamic_threshold:
                        is_speech = False
                else:
                    if frame_energy < min_energy_threshold:
                        is_speech = False
                
                speech_frames.append(is_speech)
            
            # 合并连续语音段
            segments = []
            in_speech = False
            current_segment_start = 0
            silence_duration = 0
            
            for i, is_speech in enumerate(speech_frames):
                time = i * frame_duration / 1000.0
                
                if is_speech and not in_speech:
                    in_speech = True
                    current_segment_start = time
                    silence_duration = 0
                elif not is_speech and in_speech:
                    silence_duration += frame_duration / 1000.0
                    
                    if silence_duration > max_silence_duration:
                        segment_end = time - silence_duration
                        segment_duration = segment_end - current_segment_start
                        
                        if segment_duration >= min_speech_duration:
                            if segment_duration > max_segment_duration:
                                # 分割过长段落
                                num_subsegments = int(np.ceil(segment_duration / max_segment_duration))
                                subsegment_duration = segment_duration / num_subsegments
                                
                                for j in range(num_subsegments):
                                    start = current_segment_start + j * subsegment_duration
                                    end = start + subsegment_duration
                                    segments.append((start, end))
                            else:
                                segments.append((current_segment_start, segment_end))
                        
                        in_speech = False
                elif not is_speech:
                    silence_duration = 0
            
            # 处理最后一个语音段
            if in_speech:
                end_time = audio_length / sample_rate
                segment_duration = end_time - current_segment_start
                
                if segment_duration >= min_speech_duration:
                    if segment_duration > max_segment_duration:
                        num_subsegments = int(np.ceil(segment_duration / max_segment_duration))
                        subsegment_duration = segment_duration / num_subsegments
                        
                        for j in range(num_subsegments):
                            start = current_segment_start + j * subsegment_duration
                            end = start + subsegment_duration
                            segments.append((start, end))
                    else:
                        segments.append((current_segment_start, end_time))
            
            # 确保所有片段在音频范围内
            total_duration = audio_length / sample_rate
            filtered_segments = []
            for start, end in segments:
                start = max(0, start)
                end = min(end, total_duration)
                if start < end:
                    filtered_segments.append((start, end))
            
            return filtered_segments
        
        except Exception as e:
            logger.error(f"VAD检测过程中发生错误: {str(e)}")
            return []
    
    def handle_invalid_input(self) -> List[Dict]:
        """处理无效输入，返回空音频片段列表"""
        try:
            empty_waveform = torch.zeros(1, 1, 1)
            return [{"waveform": empty_waveform, "sample_rate": 44100}]
        except:
            return [{"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}]
