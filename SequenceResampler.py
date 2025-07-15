import os
import glob
import shutil
import time

def log(msg, tag="INFO"):
    pass

def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def get_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        return []
    files = []
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            files.append(file_path)
    return sorted(files)

def safe_copy_file(source_file, destination_path):
    """安全复制文件，处理权限问题"""
    try:
        # 如果目标文件已存在且被锁定，先尝试删除
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except PermissionError:
                # 如果删除失败，尝试重命名旧文件
                timestamp = int(time.time())
                os.rename(destination_path, f"{destination_path}.old_{timestamp}")
        shutil.copy2(source_file, destination_path)
        return True
    except Exception as e:
        log(f"Error copying {source_file} to {destination_path}: {e}")
        return False

def resample_sequence(imageDirectory, batchSize, overlapSize):
    # 准备输出目录 - 使用更安全的命名
    timestamp = int(time.time())
    output_dir = os.path.join(imageDirectory, f"resampled_output_{timestamp}")
    
    # 确保输出目录存在
    create_folder(output_dir)
    
    # 获取并排序输入文件
    all_files = get_files_in_folder(imageDirectory)
    total_files = len(all_files)
    
    if total_files == 0:
        return "", "No files found in input directory"
    
    # 计算实际保留的文件索引
    selected_indices = []
    step = batchSize - overlapSize  # 有效步长
    
    if step <= 0:
        return "", "ERROR: OverlapSize must be smaller than BatchSize"
    
    current_index = 0
    while current_index < total_files:
        # 计算当前批次的结束位置
        end_index = min(current_index + batchSize - overlapSize, total_files)
        
        # 添加当前批次（不含重叠部分）
        for i in range(current_index, end_index):
            selected_indices.append(i)
        
        # 移动到下一个批次起点（跳过重叠部分）
        current_index += batchSize
    
    # 复制选中的文件到输出目录
    copy_success = 0
    copy_fail = 0
    for idx in selected_indices:
        source_file = all_files[idx]
        file_name = os.path.basename(source_file)
        destination_path = os.path.join(output_dir, file_name)
        
        if safe_copy_file(source_file, destination_path):
            copy_success += 1
        else:
            copy_fail += 1
    
    # 准备日志信息
    log_lines = [
        f"输入目录: {imageDirectory}",
        f"原始图像数量: {total_files}",
        f"批次大小: {batchSize}, 重叠大小: {overlapSize}",
        f"有效步长: {step}",
        f"重组后图像数量: {len(selected_indices)}",
        f"成功复制: {copy_success}, 失败: {copy_fail}",
        f"输出目录: {output_dir}"
    ]
    
    return output_dir, "\n".join(log_lines)

class SequenceResampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputDirectory": ("STRING", {"default": "X:/your/image/folder"}),
                "BatchSize": ("INT", {"default": 25, "min": 2, "max": 9999}),
                "OverlapSize": ("INT", {"default": 5, "min": 1, "max": 9999}),
            },
        }

    FUNCTION = "resample"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("OutputDirectory", "Log")
    CATEGORY = "Sequence Processing"

    def resample(self, inputDirectory, BatchSize, OverlapSize):
        return resample_sequence(inputDirectory, BatchSize, OverlapSize)