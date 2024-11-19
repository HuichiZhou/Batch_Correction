import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_path: str = ""):
        # 定义文件夹到标签的映射，共 51 个文件夹，对应标签从 0 到 50
        self.class_map = {
            'HEPG2-01': 0, 'HEPG2-02': 1, 'HEPG2-03': 2, 'HEPG2-04': 3, 'HEPG2-05': 4, 'HEPG2-06': 5, 'HEPG2-07': 6,
            'HEPG2-08': 7, 'HEPG2-09': 8, 'HEPG2-10': 9, 'HEPG2-11': 10, 'HUVEC-01': 11, 'HUVEC-02': 12, 'HUVEC-03': 13,
            'HUVEC-04': 14, 'HUVEC-05': 15, 'HUVEC-06': 16, 'HUVEC-07': 17, 'HUVEC-08': 18, 'HUVEC-09': 19, 'HUVEC-10': 20,
            'HUVEC-11': 21, 'HUVEC-12': 22, 'HUVEC-13': 23, 'HUVEC-14': 24, 'HUVEC-15': 25, 'HUVEC-16': 26, 'HUVEC-17': 27,
            'HUVEC-18': 28, 'HUVEC-19': 29, 'HUVEC-20': 30, 'HUVEC-21': 31, 'HUVEC-22': 32, 'HUVEC-23': 33, 'HUVEC-24': 34,
            'RPE-01': 35, 'RPE-02': 36, 'RPE-03': 37, 'RPE-04': 38, 'RPE-05': 39, 'RPE-06': 40, 'RPE-07': 41, 'RPE-08': 42,
            'RPE-09': 43, 'RPE-10': 44, 'RPE-11': 45, 'U2OS-01': 46, 'U2OS-02': 47, 'U2OS-03': 48, 'U2OS-04': 49, 'U2OS-05': 50
        }
        self.data_path = data_path
        self.data = []
        
        # 遍历每个文件夹，读取所有图像路径
        for folder_name, label in self.class_map.items():
            folder_path = os.path.join(self.data_path, folder_name)
            for filepath in glob.iglob(os.path.join(folder_path, '*.png')):
                self.data.append([filepath, label])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        # 使用 PIL 读取图像并转换为 NumPy 数组
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 确保图像为 (512, 512, 6) 的形状
        if image.shape != (512, 512, 3):
            raise ValueError(f"Unexpected image shape: {image.shape}, expected (512, 512, 6)")
        
        # 将图像转换为 PyTorch 张量并调整维度顺序 (C, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 对图像进行归一化处理
        mean = img_tensor.mean(dim=(1, 2), keepdim=True)
        var = img_tensor.var(dim=(1, 2), keepdim=True)
        img_tensor = (img_tensor - mean) / (var + 1.e-6)**.5
        
        # 将类别标签转换为张量

        # class_id = torch.tensor([class_id])
        class_id_one_hot = torch.zeros(len(self.class_map),dtype=torch.float)
        class_id_one_hot[class_id] = 1.0
        
        return img_tensor, class_id_one_hot

# # 示例用法
# dataset = CustomDataset(data_path="/home/gyang/MAE-GAN/test_image")
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # 获取前两个 batch 的数据
# for i, (images, labels) in enumerate(dataloader):
#     print(f"Batch {i + 1}:")
#     print(f"Images: {images.shape}")

#     image = images[0]
#     # 获取三个通道
#     channel_0 = image[0]  # 第一个通道
#     channel_1 = image[1]  # 第二个通道
#     channel_2 = image[2]  # 第三个通道
    
#     # 比较三个通道是否相等
#     if torch.equal(channel_0, channel_1) and torch.equal(channel_1, channel_2):
#         print(f"Image {i} has identical values across all three channels.")
#     else:
#         print(f"Image {i} has different values in its channels.")    
#     print(f"Labels: {labels}")
    
#     # 只打印前两个 batch
#     if i >= 0:
        # break
