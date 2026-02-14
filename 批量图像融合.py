import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def detect_black_edges(img, black_threshold=10, min_neighbors=1):
    """黑色边缘检测"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, binary = cv2.threshold(gray, black_threshold, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neighbor_count = cv2.filter2D(cleaned, cv2.CV_16U, neighbor_kernel) // 255
    edge_mask = (cleaned == 255) & (neighbor_count >= min_neighbors)

    return edge_mask.astype(np.uint8) * 255

def v2_common_mask(img):
    # 确保灰度
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（只用于背景判定）
    _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    h, w = bw.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # flood fill 背景白--> remove background
    ff = bw.copy()
    cv2.floodFill(ff, mask, (0, 0), 128)

    # 背景区域
    background = (ff == 128)

    # ⭐ 关键：结果来自原图
    result = img.copy()
    result[background] = 0  # 仅修改背景

    return result


def process_pair(lq_path, gt_path, lq_output_dir, gt_output_dir, lq_bar_path, gt_bar_path):
    """处理单对图像并保存结果"""
    lq = cv2.imread(str(lq_path))
    gt = cv2.imread(str(gt_path))
    lq_bar_v2 = cv2.imread(str(lq_bar_path)) if lq_bar_path is not None else None
    gt_bar_v2 = cv2.imread(str(gt_bar_path)) if gt_bar_path is not None else None

    if lq is None or gt is None:
        print(f"警告：无法读取 {lq_path.name} 或 {gt_path.name}")
        return

    # 执行边缘检测和融合
    lq_edges = detect_black_edges(lq)
    gt_edges = detect_black_edges(gt)
    common_mask = lq_edges + gt_edges

    #v2.0版本的IP策略中用以消除背景噪声的函数，本质是优化CM的计算，此处实现模块1
    common_mask = v2_common_mask(common_mask)

    # 应用融合
    common_mask = common_mask.astype(np.float32) / 255.0
    common_mask = np.expand_dims(common_mask, axis=-1)  # 将形状从 (512, 512) 改为 (512, 512, 1)
    if lq_bar_v2 is None and gt_bar_v2 is None:  #如果没有按照IP_V2进行预处理,则继续按照IP_V1进行处理
        gt_bar = common_mask * lq + (1 - common_mask) * gt
        lq_bar = common_mask * gt + (1 - common_mask) * lq
        # 保存结果（保持原始文件名）
        gt_bar_path = os.path.join(gt_output_dir, gt_path.name)
        lq_bar_path = os.path.join(lq_output_dir, lq_path.name)
        cv2.imwrite(gt_bar_path, gt_bar)
        cv2.imwrite(lq_bar_path, lq_bar)

    if lq_bar_v2 is not None:   #add noise处理后的文件夹非空则保存lq_bar
        lq_bar = common_mask * lq_bar_v2 + (1 - common_mask) * lq
        lq_bar_path = os.path.join(lq_output_dir, lq_path.name)
        cv2.imwrite(lq_bar_path, lq_bar)

    if gt_bar_v2 is not None:   #remove noise处理后的文件夹非空则保存gt_bar
        gt_bar = common_mask * gt_bar_v2 + (1 - common_mask) * gt
        gt_bar_path = os.path.join(gt_output_dir, gt_path.name)
        cv2.imwrite(gt_bar_path, gt_bar)

def batch_process(input_dir, output_dir, lq_bar_dir, gt_bar_dir):
    """批量处理目录中的所有图像对"""
    lq_dir = os.path.join(input_dir, "lq")
    gt_dir = os.path.join(input_dir, "gt")

    # 创建输出目录
    lq_output_dir = os.path.join(output_dir, "lq_bar")
    gt_output_dir = os.path.join(output_dir, "gt_bar")
    os.makedirs(lq_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    # 获取配对文件列表
    lq_bar_dir = lq_bar_dir or None
    gt_bar_dir = gt_bar_dir or None
    lq_paths = sorted(Path(lq_dir).glob("*.png"))
    gt_paths = sorted(Path(gt_dir).glob("*.png"))
    if lq_bar_dir is not None:
        lq_bar_paths = sorted(Path(lq_bar_dir).glob("*.png"))
        assert [p.name for p in lq_paths] == [p.name for p in lq_bar_paths], "LQ和add noise后的NDCT文件名不匹配"
    else:
        lq_bar_paths = None

    if gt_bar_dir is not None:
        gt_bar_paths = sorted(Path(gt_bar_dir).glob("*.png"))
        assert [p.name for p in lq_paths] == [p.name for p in gt_bar_paths], "LQ和remove noise后的uLDCT文件名不匹配"
    else:
        gt_bar_paths = None


    # 验证文件名匹配
    assert [p.name for p in lq_paths] == [p.name for p in gt_paths], "LQ和GT文件名不匹配"

    # 处理所有图像对
    if lq_bar_paths is not None and gt_bar_paths is not None:
        iterator = zip(lq_paths, gt_paths, lq_bar_paths, gt_bar_paths)
    elif lq_bar_paths is not None:
        iterator = zip(lq_paths, gt_paths, lq_bar_paths, [None] * len(lq_paths))
    elif gt_bar_paths is not None:
        iterator = zip(lq_paths, gt_paths, [None] * len(lq_paths), gt_bar_paths)
    else:
        iterator = zip(lq_paths, gt_paths, [None] * len(lq_paths), [None] * len(lq_paths))

    for lq_path, gt_path, lq_bar_path, gt_bar_path in tqdm(iterator, total=len(lq_paths)):
        process_pair(
            lq_path,
            gt_path,
            lq_output_dir,
            gt_output_dir,
            lq_bar_path,
            gt_bar_path
        )


if __name__ == '__main__':
    # 配置路径（请修改为实际路径）
    input_directory = "C:/Users/pytorch/Desktop/Dataset/dataset/train"  # 包含lq和gt子目录的目录
    add_noise_directory = 'C:/Users/pytorch/Desktop/Dataset/dataset/simulated/2percent/tradition'
    remove_noise_directory = ''
    output_directory = "C:/Users/pytorch/Desktop/Dataset/dataset/IP_v2/train/2"  # 结果保存目录

    # 执行批量处理
    print("开始批量处理...")
    batch_process(input_directory, output_directory, add_noise_directory, remove_noise_directory)
    print(f"处理完成！结果已保存到 {output_directory}")