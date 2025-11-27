# 序列重采样节点
import os
import shutil
import time
import glob

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
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except PermissionError:
                timestamp = int(time.time())
                os.rename(destination_path, f"{destination_path}.old_{timestamp}")
        shutil.copy2(source_file, destination_path)
        return True
    except Exception as e:
        log(f"Error copying {source_file} to {destination_path}: {e}")
        return False

def resample_sequence(imageDirectory, batchSize, Lapsize):
    # 准备输出目录
    timestamp = int(time.time())
    output_dir = os.path.join(imageDirectory, f"resampled_output_{timestamp}")
    create_folder(output_dir)
    
    # 获取并排序输入文件
    all_files = get_files_in_folder(imageDirectory)
    total_files = len(all_files)
    
    if total_files == 0:
        return output_dir, "No files found in input directory"
    
    if Lapsize >= batchSize:
        return output_dir, "ERROR: Lapsize must be smaller than BatchSize"
    
    # 图像序列重组逻辑
    selected_indices = []
    block_size = batchSize + Lapsize  # 完整块大小
    
    # 处理第一个块（完整保留）
    first_block_end = min(block_size, total_files)
    for i in range(0, first_block_end):
        selected_indices.append(i)
    
    # 后续块处理
    current_block_start = block_size
    block_count = 1
    
    while current_block_start < total_files:
        # 跳过前Lapsize张
        keep_start = current_block_start + Lapsize
        
        # 检查是否有足够的图像可以保留
        if keep_start >= total_files:
            break
        
        # 计算保留结束位置
        keep_end = min(keep_start + batchSize, total_files)
        
        # 添加当前块要保留的部分
        for i in range(keep_start, keep_end):
            selected_indices.append(i)
        
        # 移动到下一个块的起始位置
        current_block_start += block_size
        block_count += 1
    
    # 复制选中的文件到输出目录，并按照0000格式命名
    copy_success = 0
    copy_fail = 0
    renamed_files = []
    
    # 确定新文件名所需的位数
    total_selected = len(selected_indices)
    if total_selected == 0:
        num_digits = 4
    else:
        num_digits = len(str(total_selected - 1))
        num_digits = max(4, num_digits)  # 确保至少4位
    
    for idx, file_index in enumerate(selected_indices):
        source_file = all_files[file_index]
        file_name = os.path.basename(source_file)
        file_ext = os.path.splitext(file_name)[1]
        
        # 生成新的文件名，格式为0000、0001等
        new_file_name = f"{idx:0{num_digits}d}{file_ext}"
        destination_path = os.path.join(output_dir, new_file_name)
        
        if safe_copy_file(source_file, destination_path):
            copy_success += 1
            renamed_files.append((file_name, new_file_name))
        else:
            copy_fail += 1
    
    # 准备日志信息
    log_lines = [
        f"输入目录: {imageDirectory}",
        f"原始图像数量: {total_files}",
        f"批次大小: {batchSize}, 重叠大小: {Lapsize}",
        f"块大小: {block_size}",
        f"重组后图像数量: {len(selected_indices)}",
        f"成功复制: {copy_success}, 失败: {copy_fail}",
        f"输出目录: {output_dir}",
        f"新文件名格式: {num_digits}位数字 (例如: {renamed_files[0][1] if renamed_files else 'N/A'} 到 {renamed_files[-1][1] if renamed_files else 'N/A'})"
    ]
    
    if selected_indices:
        log_lines.append(f"保留范围: {selected_indices[0]}-{selected_indices[-1]}")
    else:
        log_lines.append("无文件保留")
    
    # 添加文件重命名示例
    if renamed_files:
        example_text = f"重命名示例: {renamed_files[0][0]} -> {renamed_files[0][1]}"
        if len(renamed_files) > 1:
            example_text += f", {renamed_files[-1][0]} -> {renamed_files[-1][1]}"
        log_lines.append(example_text)
    
    return output_dir, "\n".join(log_lines)

class SequenceResampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputDirectory": ("STRING", {"default": "X:/your/image/folder", "tooltip": "输入目录"}),
                "BatchSize": ("INT", {"default": 25, "min": 2, "max": 9999, "tooltip": "批次大小"}),
                "Lapsize": ("INT", {"default": 5, "min": 1, "max": 9999, "tooltip": "重叠大小"}),
            },
        }

    FUNCTION = "resample"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("输出目录", "日志")
    CATEGORY = "Sequence Processing"

    def resample(self, inputDirectory, BatchSize, Lapsize):
        return resample_sequence(inputDirectory, BatchSize, Lapsize)
