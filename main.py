# Загрузка библиотек
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import gdown
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import time

# Загрузка датасета
gdown.download('https://storage.yandexcloud.net/algorithmic-datasets/bus.zip', None, quiet=True)
!unzip -q "bus.zip" -d /content/bus

# Путь к данным
IMAGE_PATH = '/content/bus/'

# Получение списка классов
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
CLASS_COUNT = len(CLASS_LIST)
print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')

# Создание списков файлов и меток
data_files = []
data_labels = []

for class_label, class_name in enumerate(CLASS_LIST):
    class_path = os.path.join(IMAGE_PATH, class_name)
    class_files = os.listdir(class_path)
    print(f'Размер класса {class_name} составляет {len(class_files)} фото')
    
    for file_name in class_files:
        data_files.append(os.path.join(class_path, file_name))
        data_labels.append(class_label)

print('\nОбщий размер базы для обучения:', len(data_labels))

# Визуализация примеров изображений
fig, axs = plt.subplots(1, CLASS_COUNT, figsize=(10, 5))
for i, class_name in enumerate(CLASS_LIST):
    class_files = [f for f in data_files if class_name in f]
    img_path = random.choice(class_files)
    axs[i].set_title(class_name)
    axs[i].imshow(Image.open(img_path))
    axs[i].axis('off')
plt.show()

# Разделение данных на тренировочную и валидационную выборки
X_train_files, X_val_files, y_train_labels, y_val_labels = train_test_split(
    data_files, data_labels, test_size=0.2, random_state=42, stratify=data_labels
)

# Проверка распределения классов
print("\nРаспределение классов в обучающей выборке:")
print(np.unique(y_train_labels, return_counts=True))
print("Распределение классов в валидационной выборке:")
print(np.unique(y_val_labels, return_counts=True))

# Функция для загрузки и предобработки изображений
def load_and_preprocess_image(path, target_size=(128, 128)):
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Нормализация пикселей
    return img_array

# Загрузка изображений
print("\nЗагрузка тренировочных изображений...")
X_train = np.array([load_and_preprocess_image(path) for path in X_train_files])
print("Загрузка валидационных изображений...")
X_val = np.array([load_and_preprocess_image(path) for path in X_val_files])

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train_labels, num_classes=CLASS_COUNT)
y_val = to_categorical(y_val_labels, num_classes=CLASS_COUNT)

# Проверка форм данных
print("\nФорма данных:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# Создание модели CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(CLASS_COUNT, activation='softmax')
])

# Компиляция модели
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Обучение модели
print("\nНачало обучения модели...")
start_time = time.time()

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

end_time = time.time()
print(f"\nОбучение заняло {(end_time - start_time)/60:.2f} минут")

# Оценка модели
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nРезультат на валидационной выборке:")
print(f"Потери: {val_loss:.4f}")
print(f"Точность: {val_acc*100:.2f}%")

# Визуализация процесса обучения
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('График точности')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('График потерь')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Сохранение модели
model.save('bus_passengers_classifier.h5')
print("\nМодель сохранена как 'bus_passengers_classifier.h5'")

# Тестирование на случайных изображениях из валидационного набора
def predict_random_validation_images(num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        idx = random.randint(0, len(X_val) - 1)
        img = X_val[idx]
        true_label = CLASS_LIST[y_val_labels[idx]]
        
        # Предсказание
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        pred_label = CLASS_LIST[np.argmax(pred)]
        confidence = np.max(pred)
        
        # Отображение
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nПримеры предсказаний на валидационных данных:")
predict_random_validation_images()