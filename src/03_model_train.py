"""
03_model_train.py
--------------------
بناء وتدريب نموذج التصنيف باستخدام Transfer Learning (MobileNetV2).
هذا الخيار مناسب لأنه:
- خفيف ومناسب لأجهزة عادية / Colab المجاني
- يحتاج بيانات تدريب أقل من بناء CNN من الصفر
- دقة جيدة على مهام تصنيف الصور الطبية بعد Fine-tuning
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np

IMG_SIZE = (224, 224, 3)


def build_model(num_classes: int, fine_tune_base: bool = False) -> Model:
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE
    )
    base_model.trainable = fine_tune_base  # مجمّد في البداية لتسريع التدريب

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    حساب أوزان الفئات لمعالجة عدم توازن البيانات
    (مثلاً: nv لديها آلاف الصور بينما df قد لا تتجاوز المئات)
    """
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return dict(zip(classes, weights))


def train_model(model: Model, train_gen, val_gen, epochs=20, class_weights=None):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(
            "../models/best_model.h5", monitor="val_accuracy", save_best_only=True
        ),
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    return history


if __name__ == "__main__":
    print(
        "هذا السكربت يُستخدم بعد تجهيز train_gen و val_gen من ملف "
        "02_preprocessing.py داخل بيئة Colab حيث تتوفر بيانات HAM10000."
    )
