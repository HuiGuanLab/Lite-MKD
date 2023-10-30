import os
import cv2
import matplotlib.pyplot as plt

# 定义动作类别和视频名称
dataset=['kinetics','ucf','ucf','kinetics',]
action_categories = ["dancing_ballet","PommelHorse","GolfSwing", "dancing_macarena"]
video_names = ["_2gDwnq9OqA_000079_000089","v_PommelHorse_g01_c01","v_GolfSwing_g01_c03", "1-tbCY0OyZE_000193_000203"]



# 定义模态
modalities = ['rgb', 'depth', 'flow']

# 定义图片的基本路径
base_dir = "imp_datasets/video_datasets/data"

# 创建5x3子图
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6, 8))

# 按照要求生成图片
for i, (dataset,action_category, video_name) in enumerate(zip(dataset,action_categories, video_names)):
    for j, modality in enumerate(modalities):
        # 构建每个图片的路径
        img_path = os.path.join(base_dir, dataset,f'{modality}_l8',action_category, video_name)
        img_name = sorted(os.listdir(img_path))[0]  # 获取目录下的第一张图片
        img_path = os.path.join(img_path, img_name)

        # 使用opencv读取并resize图片
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))

        # 将图片展示在子图上
        axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # axes[i, j].set_title(f'{action_category} - {modality}', fontsize=8)
        axes[i, j].axis('off')

# 展示所有图片
axes[0, 0].set_title('RGB', fontsize=12)
axes[0, 1].set_title('Depth', fontsize=12)
axes[0, 2].set_title('Flow', fontsize=12)
plt.tight_layout()
plt.savefig('multi_modality.pdf', DPI=300)
plt.show()

