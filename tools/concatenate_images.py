from PIL import Image

def concatenate_images(img1, img2, output_path=None, background=(255, 255, 255)):
    """
    将两张图片左右拼接，顶部对齐，高度不一致时自动填充背景
    
    :param img1_path: 第一张图片路径
    :param img2_path: 第二张图片路径
    :param output_path: 输出路径（可选）
    :param background: 填充背景颜色，默认为白色
    :return: 拼接后的Image对象
    """
    
    # 获取图片尺寸
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # 计算拼接后的尺寸
    total_width = w1 + w2
    max_height = max(h1, h2)
    
    # 处理第一张图片（如果需要填充高度）
    if h1 < max_height:
        new_img1 = Image.new('RGB', (w1, max_height), background)
        new_img1.paste(img1, (0, 0))
        img1 = new_img1
    
    # 处理第二张图片（如果需要填充高度）
    if h2 < max_height:
        new_img2 = Image.new('RGB', (w2, max_height), background)
        new_img2.paste(img2, (0, 0))
        img2 = new_img2
    
    # 创建拼接后的画布
    concatenated = Image.new('RGB', (total_width, max_height), background)
    concatenated.paste(img1, (0, 0))
    concatenated.paste(img2, (w1, 0))
    
    # 保存结果（如果指定了输出路径）
    if output_path:
        concatenated.save(output_path)
    
    return concatenated