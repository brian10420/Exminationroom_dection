from PIL import Image
import os

def resize_images(source_dir, target_dir, size=(256, 256)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    images = [img for img in os.listdir(source_dir) if img.endswith(('.png', '.jpg'))]
    for img_name in images:
        img_path = os.path.join(source_dir, img_name)
        img = Image.open(img_path)
        img_resized = img.resize(size, Image.Resampling.LANCZOS)  # 使用高质量的图像缩放算法
        img_resized.save(os.path.join(target_dir, img_name))

# 调整原始图像和掩码图像的大小
resize_images(r'D:\NCUE_lab\sementic_segmentation\CamVid\images_5000', r'D:\NCUE_lab\Vision\Train_ori')
resize_images(r'D:\NCUE_lab\sementic_segmentation\CamVid\images_hand_5000', r'D:\NCUE_lab\Vision\Train_Label', size=(256, 256))
