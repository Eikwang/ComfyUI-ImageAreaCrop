# 视频帧数计算节点
from .common_imports import os, cv2


class VideoFrameCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "tooltip": "视频文件路径"}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("总帧数",)
    FUNCTION = "calculate_frames"
    CATEGORY = "video_utils"

    def clean_path(self, path):
        """清理文件路径"""
        cleaned = path.strip().strip('"').strip()
        cleaned = cleaned.replace('\\\\', '\\')
        return cleaned

    def calculate_frames(self, video_path):
        """计算视频总帧数"""
        # 清理路径
        clean_path = self.clean_path(video_path)
        
        # 验证路径有效性
        if not os.path.isfile(clean_path):
            raise ValueError(f"视频文件不存在: {clean_path}")
        
        # 尝试使用OpenCV获取视频信息
        try:
            cap = cv2.VideoCapture(clean_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {clean_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            
            # 通过时长和帧率计算总帧数（更可靠的方式）
            total_frames = int(duration * fps)
            
        finally:
            cap.release()
        
        # 二次验证计算结果
        if fps <= 0 or total_frames <= 0:
            raise ValueError("无法获取有效的视频参数")
            
        return (total_frames,)
