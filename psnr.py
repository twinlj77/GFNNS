import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
import os

def calculate_psnr(img1, img2):
    """
    计算两张图片的PSNR值。

    参数:
        img1 (np.ndarray): 第一张图片，numpy数组形式，数据类型应为float32，值域[0, 1]。
        img2 (np.ndarray): 第二张图片，numpy数组形式，数据类型应为float32，值域[0, 1]。

    返回:
        float: PSNR值。
    """
    # 确保两张图片具有相同的尺寸
    assert img1.shape == img2.shape, "Images must have the same dimensions."

    # 计算PSNR
    psnr = compare_psnr(img1, img2, data_range=1.0)
    return psnr


def load_image_as_float(image_path):
    """
    加载图像并将其转换为float32类型，值域[0, 1]。

    参数:
        image_path (str): 图像文件的路径。

    返回:
        np.ndarray: 加载后的图像，numpy数组形式。
    """
    img = Image.open(image_path).convert('RGB')  # 转换为RGB模式
    img = np.array(img, dtype=np.float32) / 255.0  # 转换为float32并归一化到[0, 1]
    return img


# 主函数：计算一张图片与一组图片的PSNR值，并输出平均值
def main():
    # 加载基准图像
    base_image_path = '下载 (1).jpg'  # 替换为你的基准图像路径
    base_image = load_image_as_float(base_image_path)

    # 设置比较图像的文件夹路径
    compare_images_folder = 'compare'  # 替换为你的比较图像文件夹路径
    compare_image_files = [f for f in os.listdir(compare_images_folder) if f.endswith('.jpg')]  # 假设图片格式为.jpg

    # 初始化PSNR值的总和和计数器
    total_psnr = 0.0
    num_images = len(compare_image_files)

    # 计算PSNR值
    for compare_image_file in compare_image_files:
        compare_image_path = os.path.join(compare_images_folder, compare_image_file)
        compare_image = load_image_as_float(compare_image_path)
        psnr_value = calculate_psnr(base_image, compare_image)
        total_psnr += psnr_value
        #print(f"PSNR for {compare_image_file}: {psnr_value:.2f}")

        # 计算平均PSNR值并输出
    average_psnr = total_psnr / num_images
    print(f"Average PSNR: {average_psnr:.2f}")


if __name__ == "__main__":
    main()