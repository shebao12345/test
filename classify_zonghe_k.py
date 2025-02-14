import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# 训练集文件夹路径
train_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train_small"
# 测试集文件夹路径
test_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train_small"

# 统一图像尺寸
target_size = (224, 224)

# 提取颜色直方图特征
def extract_color_histogram_feature(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 提取CLD颜色特征
def extract_cld_feature(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    num_subregions_x = 3
    num_subregions_y = 3
    height, width, _ = hsv_image.shape
    subregion_height = height // num_subregions_y
    subregion_width = width // num_subregions_x
    cld_feature_vector = []

    for i in range(num_subregions_y):
        for j in range(num_subregions_x):
            y_start = i * subregion_height
            y_end = (i + 1) * subregion_height
            x_start = j * subregion_width
            x_end = (j + 1) * subregion_width
            subregion_image = hsv_image[y_start:y_end, x_start:x_end]
            hist = cv2.calcHist([subregion_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            cld_feature_vector.extend(hist.flatten())

    return np.array(cld_feature_vector)

# 提取SCD（Scalable Color Descriptor）颜色特征
def extract_scd_feature(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_bins = 16
    s_bins = 4
    v_bins = 4
    hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hsv_hist, hsv_hist)
    return hsv_hist.flatten()

# 加载图像数据并提取多种特征
def load_data_and_extract_features(folder_path):
    data = []
    labels = []
    images = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            # 调整图像尺寸
            image = cv2.resize(image, target_size)
            images.append(image)

            color_histogram_feature = extract_color_histogram_feature(image)
            cld_feature = extract_cld_feature(image)
            scd_feature = extract_scd_feature(image)

            combined_feature = np.concatenate((color_histogram_feature, cld_feature, scd_feature))

            data.append(combined_feature)
            labels.append(class_folder)

    return np.array(data), np.array(labels), np.array(images)

# 构建CNN模型进行特征提取
base_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten()
])

# 加载训练集数据并提取多种特征
train_handcrafted_features, train_labels, train_images = load_data_and_extract_features(train_folder_path)

# 提取CNN特征
train_cnn_features = base_model.predict(train_images)

# 拼接手工特征和CNN特征
train_combined_features = np.hstack((train_handcrafted_features, train_cnn_features))

# 对特征数据进行标准化（可选，但对于一些分类器可能有帮助）
scaler = StandardScaler()
train_combined_features = scaler.fit_transform(train_combined_features)

# 创建并训练k近邻分类器
k = 6  # 可根据需要调整k值
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(train_combined_features, train_labels)

# 加载测试集数据并提取多种特征
test_handcrafted_features, test_labels, test_images = load_data_and_extract_features(test_folder_path)

# 提取测试集的CNN特征
test_cnn_features = base_model.predict(test_images)

# 拼接测试集的手工特征和CNN特征
test_combined_features = np.hstack((test_handcrafted_features, test_cnn_features))

# 对测试集特征进行标准化
test_combined_features = scaler.transform(test_combined_features)

# 在测试集上进行预测
predicted_labels = classifier.predict(test_combined_features)

# 计算分类精度
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"分类精度: {accuracy}")