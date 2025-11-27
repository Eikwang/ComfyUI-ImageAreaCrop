# 公共导入文件 - 包含所有节点共用的导入和依赖

# 基础Python库
import torch
import numpy as np
import json
import cv2
import os
import time
import math
import io
import sys
import tempfile
import hashlib
import logging
import glob
import shutil
from typing import List, Dict, Tuple, Optional, Any

# PyTorch相关
from torchvision.transforms.functional import resize as torch_resize
from torch.nn.functional import interpolate

# 图像处理
from PIL import Image

# 音频处理
import librosa
import pyaudio
import wave
import audioop
import torchaudio
from torchaudio.transforms import Fade
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import webrtcvad

# ComfyUI相关
import comfy
from comfy.sd import VAE
from comfy import model_management
import comfy.utils
import comfy.sd
import comfy.samplers
import nodes
import folder_paths

# 其他工具
import ffmpeg
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AudioSpeechSegmenter")