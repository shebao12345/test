import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2

# 训练集文件夹路径
train_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train"
# 测试集文件夹路径
test_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\test"

# 提取CLD颜色特征
def extract_cld_feature(image):
    # 将图像转换为HSV颜色空间，以便更好地处理颜色信息
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义将图像划分的子区域数量（这里以3x3为例，可以根据需要调整）
    num_subregions_x = 3
    num_subregions_y = 3
    # 获取图像的高度和宽度
    height, width, _ = hsv_image.shape
    # 计算每个子区域的高度和宽度
    subregion_height = height // num_subregions_y
    subregion_width = width // num_subregions_x
    cld_feature_vector = []

    for i in range(num_subregions_y):
        for j in range(num_subregions_x):
            # 确定当前子区域的坐标范围
            y_start = i * subregion_height
            y_end = (i + 1) * subregion_height
            x_start = j * subregion_width
            x_end = (j + 1) * subregion_width
            # 提取当前子区域的图像
            subregion_image = hsv_image[y_start:y_end, x_start:x_end]
            # 计算子区域的颜色直方图作为该子区域的颜色特征
            hist = cv2.calcHist([subregion_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            # 将子区域的特征向量添加到CLD特征向量中
            cld_feature_vector.extend(hist.flatten())
    return np.array(cld_feature_vector)

# 加载图像数据并提取特征
def load_data(folder_path):
    data = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            feature_vector = extract_cld_feature(image)
            data.append(feature_vector)
            labels.append(class_folder)
    return np.array(data), np.array(labels)


# 加载训练集数据
train_data, train_labels = load_data(train_folder_path)

# 对特征数据进行标准化（可选，但对于一些分类器可能有帮助）
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# 创建并训练k近邻分类器
k = 6  # 可根据需要调整k值
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(train_data, train_labels)

# 加载测试集数据
test_data, test_labels = load_data(test_folder_path)
test_data = scaler.transform(test_data)

# 在测试集上进行预测
predicted_labels = classifier.predict(test_data)

# 计算分类精度
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"分类精度: {accuracy}")