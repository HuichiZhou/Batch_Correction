import os
import shutil
import random

def copy_random_10_images(source_root, target_root):
    # 获取所有子文件夹列表
    subfolders = [f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))]

    for subfolder in subfolders:
        plate1_path = os.path.join(source_root, subfolder, 'Plate1')
        target_subfolder_path = os.path.join(target_root, subfolder)

        if os.path.exists(plate1_path):
            # 确保目标子文件夹存在
            os.makedirs(target_subfolder_path, exist_ok=True)

            # 获取Plate1文件夹中的所有图像文件
            images = [f for f in os.listdir(plate1_path) if os.path.isfile(os.path.join(plate1_path, f))]

            # 随机选择10个图像文件
            random_images = random.sample(images, min(10, len(images)))

            # 复制随机选择的图像文件到目标文件夹
            for image in random_images:
                source_image_path = os.path.join(plate1_path, image)
                target_image_path = os.path.join(target_subfolder_path, image)
                shutil.copy2(source_image_path, target_image_path)
                print(f'Copied {source_image_path} to {target_image_path}')

if __name__ == "__main__":
    source_root = "/media/NAS06/zhc/rxrx1/images"
    target_root = "/home/gyang/MAE-GAN/test_image"  # 修改为你想保存图像的目标路径
    copy_random_10_images(source_root, target_root)