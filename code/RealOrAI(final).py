import os # os : 운영체제(Operating System) 기능을 다루는 표준 라이브러리
os.environ['KAGGLE_CONFIG_DIR'] = r'C:\2025_Python\RealOrAI\.kaggle'
import zipfile
# with zipfile.ZipFile("real-ai-art.zip", 'r') as zip_ref:
#    zip_ref.extractall("real_ai_art")
from PIL import Image, UnidentifiedImageError
import numpy as np
from PIL import Image # PIL : 이미징 라이브러리
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_image(path, size=(128, 128)):
    try:
        img = Image.open(path)
        img = img.convert("RGB").resize(size)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"⚠️ 이미지 열기 실패: {path} → {e} → 건너뜀")
        return None

train_root = r'C:\2025_Python\RealOrAI\real_ai_art\Real_AI_SD_LD_Dataset\train'

style_folders = os.listdir(train_root) # train 폴더에 안에 있는 모든 하위 폴더 이름을 리스트로 저장 

image_paths = []
labels = []

for folder in style_folders:
    folder_path = os.path.join(train_root, folder)
    if not os.path.isdir(folder_path):
        continue  # 혹시 파일이 섞여 있다면 무시

    # 폴더 내 이미지 경로 수집
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, fname)
            image_paths.append(full_path)

            # 폴더 경로를 기준으로 라벨 지정 (대소문자 구분 없이)
            label = 1 if 'ai' in folder.lower() else 0
            labels.append(label)

from collections import Counter

# 예: AI/명화 균형 맞춰 5000개씩 자르기
real_idx = [i for i, l in enumerate(labels) if l == 0]
ai_idx = [i for i, l in enumerate(labels) if l == 1]

import random
random.shuffle(real_idx)
random.shuffle(ai_idx)

num_samples = min(len(real_idx), len(ai_idx), 5000)
chosen_idx = real_idx[:num_samples] + ai_idx[:num_samples]
random.shuffle(chosen_idx)

image_paths = [image_paths[i] for i in chosen_idx]
labels = [labels[i] for i in chosen_idx]

test_root = r'C:\2025_Python\RealOrAI\real_ai_art\Real_AI_SD_LD_Dataset\test'
test_folders = os.listdir(test_root)

test_paths = []
test_labels = []

for folder in test_folders:
    folder_path = os.path.join(test_root, folder)
    if not os.path.isdir(folder_path):
        continue

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, fname)
            test_paths.append(full_path)

            # 폴더 경로를 기준으로 라벨 지정 (대소문자 구분 없이)
            label = 1 if 'ai' in folder.lower() else 0
            test_labels.append(label)

real_idx_test = [i for i, l in enumerate(test_labels) if l == 0]
ai_idx_test = [i for i, l in enumerate(test_labels) if l == 1]

# 무작위 섞기
random.shuffle(real_idx_test)
random.shuffle(ai_idx_test)

# 샘플 수 설정 (여기선 2000개씩)
num_test_samples = min(len(real_idx_test), len(ai_idx_test), 2000)
chosen_test_idx = real_idx_test[:num_test_samples] + ai_idx_test[:num_test_samples]
random.shuffle(chosen_test_idx)

# 선택된 balanced 테스트셋 구성
test_paths = [test_paths[i] for i in chosen_test_idx]
test_labels = [test_labels[i] for i in chosen_test_idx]

# 확인
from collections import Counter

def preprocess_image(path, size=(128, 128)):
    img = Image.open(path).convert("RGB").resize(size) # RGB 형식으로 색깔 통일 및 size 통일
    return np.array(img, dtype=np.float32) / 255.0 # 정규화

X_test = []
y_test_clean = []

for p, label in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
    img = preprocess_image(p, size=(128, 128))
    if img is not None:
        X_test.append(img)
        y_test_clean.append(label)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test_clean)

X = []
y_clean = []

for p, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    img = preprocess_image(p)
    if img is not None:
        if img.shape == (128, 128, 3):
            X.append(img)
            y_clean.append(label)
        else:
            print(f"⚠️ 크기 오류 무시: {p}, shape={img.shape}")

X = np.array(X, dtype=np.float32)
y = np.array(y_clean)

X = []
y_clean = []

for p, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    try:
        img = preprocess_image(p)
        if img is not None and img.shape == (128, 128, 3):  # 모양까지 체크!
            X.append(img)
            y_clean.append(label)
    except Exception as e:
        print(f"⚠️ 예외 발생: {p} → {e} → 건너뜀")

X = np.array(X, dtype=np.float32)
y = np.array(y_clean)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)


