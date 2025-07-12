# ComfyUI-ImageAreaCrop

这是一个用于ComfyUI的图像区域裁剪插件，支持精确裁剪和恢复功能。
新增音频分割和时长转帧数节点。可以和循环工作流结合,制作数字人视频

## 功能特点
图像节点
ImageAreaCropNode (图像区域裁剪节点)
-    功能：
- 根据指定的坐标和尺寸裁剪图像，并可选择调整裁剪后图像的尺寸。
- 裁剪时已经考虑了边界情况，避免越界。
- 返回裁剪信息（JSON）用于后续恢复。
- 参数:
- X: 裁剪区域的左上角X坐标
- Y: 裁剪区域的左上角Y坐标
- Width: 裁剪区域的宽度
- Height: 裁剪区域的高度
- do_resize: 是否调整裁剪后图像的尺寸（可选，默认为False）
      
AreaCropRestoreNode (区域裁剪恢复节点)
-    功能：根据裁剪信息将裁剪后的图像恢复到目标图像中的原始位置。
- 注意：裁剪信息必须与目标图像匹配。

ImageReverseOrderNode (图像序列反转节点)
-    功能：反转图像序列的顺序，支持循环和去重。
- 使用索引操作代替实际数据复制，效率高。
- 循环模式通过构建索引序列实现。
- 参数:
- reverse: 是否反转图像序列（可选，默认为True）
- loop: 是否循环（可选，默认为False）
- loopcount: 循环次数（可选，默认为1）
- unique: 是否去重（可选，默认为False）

ImageTransferNode (图像传输节点)
-    功能：缓存图像，并可手动插入新图像。
- 支持手动插入图像到指定位置。
- 参数:
- merge_manual: 是否合并手动插入的图像（可选，默认为False）
- merge_index: 手动插入的图像插入位置（可选，默认为0）

音频分割节点
AudioSplitNode (音频分割节点)
-    功能：根据帧率和指定的分割长度（帧数）分割音频。
- 支持淡入淡出效果（淡出）。
- 参数:
- frame_rate: 音频帧率（设定1秒祯数）
- spilit_length: 分割长度（帧数）
- skip_length: 跳过长度（帧数）
- transition_length: 淡出长度（帧数）

AudioDurationToFrames (音频时长转帧数节点)
-    功能：根据音频时长和帧率计算总帧数。
- 参数:
- frame_rate: 音频帧率（设定1秒祯数）

AudioSpeechSegmenter (音频语音分割节点)
-    功能：使用WebRTC VAD检测语音段，并分割音频。
- 包含噪声抑制、重采样等预处理。
- 支持多种参数调整（最小语音时长、最大静音时长等）。
- 输出分割后的音频列表和分割信息。
- 参数:
- frame_duration: 计算长度(ms)
- min_speech_duration: 最小语音时长(s), 小于此长度的语音将被视为静音
- max_silence_duration: 最大静音时长(s), 大于此长度的静音将被视为语音分割点
- aggresive: 检测敏感度（1-3）
- max_segments_duration: 最大分段时长(s)
- min_energy_threshold: 最小能量阈值（0.000-0.100）调整建议：安静语音：0.005 ~ 0.01,嘈杂环境：0.02 ~ 0.05
- noise_level: 噪音水平,-30至-60.低于此设定值判定为噪音
- noise_suppression: 是否启用噪声抑制（可选，默认为True）
- resample: 重采样（默认为16000）
- debug_output: 是否输出调试信息（可选，默认为False）

AudioSegmentProcessor (音频片段中转)
-    功能：处理音频片段列表，可选择合并或选择指定片段，并支持保存到本地。
- 支持合并多个音频片段。
- 支持保存单个片段或全部片段。
- 参数:
- merge_audios: 是否合并音频片段（可选，默认为False）
- save_to_local: 是否保存到本地（可选，默认为False）
- select_segments: 选择要传输的片段索引列表（从0开始）
- file_name: 保存的文件名（保存在output，默认为“audio/xx"）

audiosilence (音频静音恢复节点)
-    功能：根据分割信息将音频恢复到目标音频中的原始位置。

videocount (视频计数节点)
-    功能：根据帧率计算视频总帧数。避免VHS加载长视频导致卡顿.