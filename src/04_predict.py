"""
04_predict.py (محدَّث - EfficientNetB0 + Focal Loss)
-------------------------------------------------------
تحميل النموذج المدرَّب فعلياً (EfficientNetB0) وتنفيذ تنبؤ على صورة جديدة،
ثم حفظ النتيجة في قاعدة بيانات المشروع (skin_cancer.db).

الفرق عن النسخة السابقة:
- معالجة مسبقة خاصة بـ EfficientNet (preprocess_input) بدل rescale 1/255
- 7 فئات حقيقية بدل التصنيف الثنائي المبسّط
- دالة focal loss مطلوبة عند تحميل النموذج (custom_objects)
"""

import sqlite3
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "skin_cancer.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model_efficientnet.h5")

# رابط تحميل النموذج من Google Drive (لبيئات سحابية مثل Streamlit Cloud
# التي لا تملك تخزيناً دائماً محلياً). اتركيه فارغاً "" إذا كنتِ تضعين
# الملف يدوياً بمجلد models/ على بيئة محلية عادية.
MODEL_DRIVE_FILE_ID = "1y7iWSDEnJN9oN3ZfsLUM7APY-KujMS4Z"  # ضعي هنا الجزء بين /d/ و /view من رابط Drive

IMG_SIZE = (224, 224)

# فئات HAM10000 السبع - بترتيب أبجدي، يطابق تماماً ترتيب class_indices
# الذي أنتجه Keras أثناء التدريب في دفتر Colab (flow_from_dataframe يرتّب
# الفئات أبجدياً تلقائياً عندما لا يُحدَّد ترتيب صريح)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# مستوى الخطورة لكل فئة (يطابق جدول lesion_types في قاعدة البيانات)
RISK_MAP = {
    "akiec": "medium",  # قبل سرطاني - يستدعي متابعة
    "bcc":   "high",    # سرطان خلايا قاعدية
    "bkl":   "low",     # حميد
    "df":    "low",     # حميد
    "mel":   "high",    # الميلانوما - الأخطر
    "nv":    "low",     # حميد (شامة عادية)
    "vasc":  "low",     # حميد
}


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    يجب توفير نفس دالة الخسارة المستخدمة أثناء التدريب عند تحميل النموذج،
    حتى لو لن نُدرِّب من جديد، لأن Keras يحتاجها لإعادة بناء بنية النموذج.
    """
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss


_model_cache = None


def get_model():
    """
    تحميل النموذج مرة واحدة فقط وإعادة استخدامه (تجنّباً لبطء إعادة التحميل).
    إذا لم يكن الملف موجوداً محلياً وتوفّر معرّف Drive، يُنزَّل تلقائياً
    (مهم لبيئات سحابية مثل Streamlit Cloud حيث لا يمكن رفع الملف يدوياً).
    """
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            if MODEL_DRIVE_FILE_ID:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                import gdown
                url = f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
            else:
                raise FileNotFoundError(
                    f"لم يتم العثور على النموذج بالمسار: {MODEL_PATH}\n"
                    "إما ضعي الملف يدوياً بمجلد models/، أو عبّئي "
                    "MODEL_DRIVE_FILE_ID بمعرّف ملف Google Drive."
                )
        _model_cache = load_model(
            MODEL_PATH,
            custom_objects={"focal_loss": categorical_focal_loss(gamma=2.0, alpha=0.25)},
        )
    return _model_cache


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """قراءة صورة ومعالجتها بنفس طريقة معالجة EfficientNet أثناء التدريب."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"تعذّرت قراءة الصورة: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)  # معالجة خاصة بـ EfficientNet (ليست rescale بسيطة)
    return img


def predict_image(image_path: str) -> dict:
    """
    تنفّذ تنبؤاً كاملاً على صورة، وتُرجع قاموساً بالنتيجة.
    لا تحفظ بقاعدة البيانات تلقائياً - استخدمي save_result_to_db بعدها.
    """
    model = get_model()
    img = load_and_preprocess_image(image_path)
    img_batch = np.expand_dims(img, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    risk_level = RISK_MAP[predicted_class]

    # كل الاحتمالات لكل فئة (مفيد لعرض تفصيلي بالواجهة إذا رغبتِ)
    all_probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence_score": round(confidence, 4),
        "risk_level": risk_level,
        "all_probabilities": all_probabilities,
    }


def save_result_to_db(image_id: int, result: dict, model_version_id: int) -> int:
    """يحفظ نتيجة التنبؤ بجدول diagnosis_results، ويُرجع id السجل الجديد."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO diagnosis_results
            (image_id, lesion_type_code, confidence_score, risk_level, model_version_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            image_id,
            result["predicted_class"],
            result["confidence_score"],
            result["risk_level"],
            model_version_id,
        ),
    )
    diagnosis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return diagnosis_id


def get_active_model_version_id() -> int:
    """يُرجع id أحدث إصدار نموذج مسجَّل بجدول model_versions (لاستخدامه بالحفظ)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM model_versions WHERE version_name = 'v2.0-efficientnetb0-focalloss'"
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        raise ValueError(
            "لم يتم العثور على إصدار النموذج بقاعدة البيانات. "
            "شغّلي أولاً سكربت register_model_version.py"
        )
    return row[0]


if __name__ == "__main__":
    print("استخدمي predict_image(image_path) بعد التأكد من وجود النموذج بمجلد models/")
