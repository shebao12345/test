import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from sklearn.linear_model import LogisticRegression

# 训练集文件夹路径
train_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train"
# 测试集文件夹路径
test_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\test"

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
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)

            color_histogram_feature = extract_color_histogram_feature(image)
            cld_feature = extract_cld_feature(image)
            scd_feature = extract_scd_feature(image)

            combined_feature = np.concatenate((color_histogram_feature, cld_feature, scd_feature))

            data.append(combined_feature)
            labels.append(class_folder)

    return np.array(data), np.array(labels)

# 加载训练集数据并提取多种特征
train_data, train_labels = load_data_and_extract_features(train_folder_path)

# 对特征数据进行标准化（可选，但对于一些分类器可能有帮助）
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# 创建并训练Logistic回归分类器
classifier = LogisticRegression()
classifier.fit(train_data, train_labels)

# 加载测试集数据并提取多种特征
test_data, test_labels = load_data_and_extract_features(test_folder_path)
test_data = scaler.transform(test_data)

# 在测试集上进行预测
predicted_labels = classifier.predict(test_data)

# 计算分类精度
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"分类精度: {accuracy}")