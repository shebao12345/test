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

# 提取颜色直方图特征
def extract_color_histogram_feature(image):
    # 将图像转换为HSV颜色空间，因为HSV对颜色的描述更符合人类视觉感知
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算颜色直方图
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # 归一化直方图，使其具有可比性
    cv2.normalize(hist, hist)
    return hist.flatten()

# 加载图像数据并提取特征
def load_data(folder_path):
    data = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            feature_vector = extract_color_histogram_feature(image)
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