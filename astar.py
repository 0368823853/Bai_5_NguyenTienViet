import os
import cv2
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----- 1. Phân lớp IRIS với CART và ID3 -----

# Tải và chia tập dữ liệu IRIS
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# CART Classifier (Gini Index)
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

print("IRIS - CART (Gini) Accuracy:", accuracy_score(y_test, y_pred_cart))
print("IRIS - CART (Gini) Classification Report:\n", classification_report(y_test, y_pred_cart))

# ID3 Classifier (Information Gain)
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

print("IRIS - ID3 (Information Gain) Accuracy:", accuracy_score(y_test, y_pred_id3))
print("IRIS - ID3 (Information Gain) Classification Report:\n", classification_report(y_test, y_pred_id3))

# ----- 2. Phân lớp ảnh nha khoa với CART và ID3 -----

# Đường dẫn thư mục chứa 300 ảnh nha khoa
image_folder = "images"
labels = []  # Danh sách nhãn của từng ảnh
images = []  # Danh sách ảnh đã xử lý

# Đọc và xử lý ảnh
for filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
    img_resized = cv2.resize(img, (64, 64))  # Thay đổi kích thước ảnh về 64x64
    images.append(img_resized.flatten())  # Chuyển đổi thành vector và thêm vào danh sách

    # Gán nhãn dựa trên tên ảnh
    if "class0" in filename:
        labels.append(0)  # Ví dụ cho class 0
    elif "class1" in filename:
        labels.append(1)  # Ví dụ cho class 1

X = np.array(images)
y = np.array(labels)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# CART Classifier (Gini Index)
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

print("Dental Images - CART (Gini) Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Dental Images - CART (Gini) Classification Report:\n", classification_report(y_test, y_pred_cart))

# ID3 Classifier (Information Gain)
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

print("Dental Images - ID3 (Information Gain) Accuracy:", accuracy_score(y_test, y_pred_id3))
print("Dental Images - ID3 (Information Gain) Classification Report:\n", classification_report(y_test, y_pred_id3))
