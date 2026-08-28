"""
02_preprocessing.py
---------------------
معالجة الصور قبل إدخالها للنموذج:
- تغيير الحجم (Resize)
- تطبيع القيم (Normalization)
- زيادة البيانات (Data Augmentation) لمعالجة عدم توازن الفئات
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """قراءة صورة وتحضيرها للنموذج"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"لم يتم العثور على الصورة أو تعذّرت قراءتها: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img


def get_augmentation_generator() -> ImageDataGenerator:
    """
    مولّد Augmentation لزيادة حجم بيانات التدريب وتحسين تعميم النموذج،
    خصوصاً للفئات النادرة مثل mel و df و vasc.
    """
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )


def build_dataframe_generators(train_df, val_df, images_dir, batch_size=32,
                                target_col="binary_label"):
    """
    بناء مولّدات بيانات (train/val) مباشرة من DataFrame يحتوي على
    عمود 'image_id' وعمود التصنيف target_col.
    يفترض أن أسماء الملفات هي image_id + '.jpg'
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["filename"] = train_df["image_id"] + ".jpg"
    val_df["filename"] = val_df["image_id"] + ".jpg"

    train_gen = get_augmentation_generator().flow_from_dataframe(
        dataframe=train_df,
        directory=images_dir,
        x_col="filename",
        y_col=target_col,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    # لا نطبق augmentation على بيانات التحقق، فقط تطبيع
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=images_dir,
        x_col="filename",
        y_col=target_col,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    return train_gen, val_gen
