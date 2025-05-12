import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np

# 初始化配置
ANNOTATION_FILE = '/mnt/d/tinydetection/datasets/aitod/annotations/aitodv2_trainval.json'
OUTPUT_PLOT = 'per_image_distribution.png'

# 初始化COCO API
coco = COCO(ANNOTATION_FILE)

# 获取所有图像ID
img_ids = coco.getImgIds()

# 初始化统计字典（区域: {数量: 图片数}）
region_stats = {
    'Top-Left': defaultdict(int),
    'Top-Right': defaultdict(int),
    'Bottom-Left': defaultdict(int),
    'Bottom-Right': defaultdict(int)
}

# 遍历所有图像
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    width = img_info['width']
    height = img_info['height']
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # 初始化当前图像的计数
    counts = [0, 0, 0, 0]  # TL, TR, BL, BR
    
    for ann in anns:
        x, y, w, h = ann['bbox']
        center_x = x + w/2
        center_y = y + h/2
        
        # 判断区域
        region = 0
        if center_x >= width/2:
            region += 1
        if center_y >= height/2:
            region += 2
        
        counts[region] += 1
    
    # 更新统计
    for i, region in enumerate(['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']):
        region_stats[region][counts[i]] += 1

# 打印统计结果
max_show = 1000  # 最大显示数量值
for region in region_stats:
    print(f"\n=== {region} 区域 ===")
    total_imgs = sum(region_stats[region].values())
    
    # 按数量排序并显示
    sorted_counts = sorted(region_stats[region].items(), key=lambda x: x)
    for count, num_imgs in sorted_counts:
        if count <= max_show:
            print(f"有 {count} 个物体的图片：{num_imgs} 张 ({num_imgs/total_imgs:.2%})")
    
    # 统计超过max_show的数量
    over_count = sum(v for k, v in region_stats[region].items() if k > max_show)
    if over_count > 0:
        print(f"超过 {max_show} 个物体的图片：{over_count} 张 ({over_count/total_imgs:.2%})")

# 可视化
plt.figure(figsize=(15, 10))

for i, (region, counts) in enumerate(region_stats.items(), 1):
    plt.subplot(2, 2, i)
    
    # 准备数据
    x = np.array(sorted(counts.keys()))
    y = np.array([counts[k] for k in x])
    
    # 截断显示范围
    mask = x <= max_show
    plt.bar(x[mask], y[mask], alpha=0.7)
    
    # 添加超出范围统计
    if sum(~mask) > 0:
        plt.bar(max_show+1, sum(y[~mask]), color='orange', alpha=0.7, label=f'>{max_show}')
    
    plt.title(f'{region} 物体数量分布')
    plt.xlabel('物体数量')
    plt.ylabel('图片数量')
    plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.show()

print(f"\n可视化结果已保存至：{OUTPUT_PLOT}")