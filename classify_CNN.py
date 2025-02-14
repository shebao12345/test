import os
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# 训练集文件夹路径
train_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train_small2"
# 测试集文件夹路径
test_folder_path = r"C:\Users\33052\Desktop\detr_data\subset of Ainimal-10\data\train_small2"

# 统一图像尺寸
target_size = (224, 224)

# 加载图像数据
def load_images(folder_path):
    data = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            # 调整图像尺寸
            image = cv2.resize(image, target_size)
            data.append(image)
            labels.append(class_folder)
    return np.array(data), np.array(labels)

# 加载训练集数据
train_images, train_labels = load_images(train_folder_path)
# 加载测试集数据
test_images, test_labels = load_images(test_folder_path)

# 对标签进行编码
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# 构建 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(train_labels)), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32)

# 在测试集上进行预测
predictions = model.predict(test_images)
predicted_labels_encoded = np.argmax(predictions, axis=1)

# 计算分类精度
accuracy = accuracy_score(test_labels_encoded, predicted_labels_encoded)
print(f"分类精度: {accuracy}")