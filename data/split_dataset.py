import os
import random
import shutil
from tqdm import tqdm

def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    将图像数据集划分为训练集、验证集和测试集。

    Args:
        root_dir (str): 数据集根目录，包含 'images' 和 'labels' 两个子文件夹。
        train_ratio (float): 训练集比例 (0到1之间)。
        val_ratio (float): 验证集比例 (0到1之间)。
        test_ratio (float): 测试集比例 (0到1之间)。
        random_seed (int): 随机种子，用于保证划分的可重复性。
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "训练集、验证集和测试集的比例之和必须为1.0"

    # 直接从根目录读取图片和标注文件
    image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(root_dir) if f.endswith('.txt')]

    output_train_images = os.path.join(root_dir, 'train', 'images')
    output_train_labels = os.path.join(root_dir, 'train', 'labels')
    output_val_images = os.path.join(root_dir, 'val', 'images')
    output_val_labels = os.path.join(root_dir, 'val', 'labels')
    output_test_images = os.path.join(root_dir, 'test', 'images')
    output_test_labels = os.path.join(root_dir, 'test', 'labels')

    # 创建输出文件夹
    os.makedirs(output_train_images, exist_ok=True)
    os.makedirs(output_train_labels, exist_ok=True)
    os.makedirs(output_val_images, exist_ok=True)
    os.makedirs(output_val_labels, exist_ok=True)
    os.makedirs(output_test_images, exist_ok=True)
    os.makedirs(output_test_labels, exist_ok=True)

    random.seed(random_seed)
    random.shuffle(image_files)

    num_images = len(image_files)
    train_split = int(train_ratio * num_images)
    val_split = int(val_ratio * num_images)

    train_images = image_files[:train_split]
    val_images = image_files[train_split:train_split + val_split]
    test_images = image_files[train_split + val_split:]

    def move_files(image_list, source_images_dir, source_labels_dir, dest_images_dir, dest_labels_dir):
        for image_file in tqdm(image_list, desc=f"Moving files to {os.path.basename(dest_images_dir)}"):
            name, ext = os.path.splitext(image_file)
            label_file = name + '.txt'  # 假设标注文件是 .txt 格式

            image_src = os.path.join(root_dir, image_file)
            label_src = os.path.join(root_dir, label_file)
            image_dst = os.path.join(dest_images_dir, image_file)
            label_dst = os.path.join(dest_labels_dir, label_file)

            if os.path.exists(image_src):
                shutil.copy2(image_src, image_dst)  # copy2保留元数据
            if os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)

    move_files(train_images, root_dir, root_dir, output_train_images, output_train_labels)
    move_files(val_images, root_dir, root_dir, output_val_images, output_val_labels)
    move_files(test_images, root_dir, root_dir, output_test_images, output_test_labels)

    print("数据集划分完成！")
    print(f"训练集图像数量: {len(train_images)}")
    print(f"验证集图像数量: {len(val_images)}")
    print(f"测试集图像数量: {len(test_images)}")

if __name__ == "__main__":
    dataset_root = 'lock_dataset'  # 将此路径替换为你的数据集根目录
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random_seed = 42

    split_dataset(dataset_root, train_ratio, val_ratio, test_ratio, random_seed)
