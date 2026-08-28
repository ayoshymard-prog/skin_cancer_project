"""
register_model_version.py
----------------------------
تسجيل إصدار نموذج EfficientNetB0 المدرَّب فعلياً بجدول model_versions،
بأرقام الأداء الحقيقية الناتجة من التقييم على Colab (82% دقة).

شغّلي هذا السكربت مرة واحدة بعد وضع best_model_efficientnet.h5 بمجلد models/.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "skin_cancer.db")

VERSION_NAME = "v2.0-efficientnetb0-focalloss"


def register():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # تفادي التكرار لو شُغِّل السكربت أكثر من مرة
    cursor.execute("SELECT id FROM model_versions WHERE version_name = ?", (VERSION_NAME,))
    if cursor.fetchone():
        print(f"الإصدار '{VERSION_NAME}' مسجَّل مسبقاً - لا حاجة لإعادة التسجيل.")
        conn.close()
        return

    cursor.execute(
        """
        INSERT INTO model_versions
            (version_name, architecture, training_dataset, accuracy,
             precision_score, recall_score, f1_score, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            VERSION_NAME,
            "EfficientNetB0 (Transfer Learning + Fine-Tuning) + Focal Loss",
            "HAM10000",
            0.82,   # Accuracy
            0.84,   # Weighted Avg Precision
            0.82,   # Weighted Avg Recall
            0.82,   # Weighted Avg F1-score
            "دقة 82% على بيانات التحقق (2003 صورة). نقطة ضعف موثّقة: "
            "Recall منخفض لفئة akiec (0.14) بسبب تشابهها البصري مع bkl وmel "
            "وقلة تمثيلها بالبيانات (3.3% فقط). راجع الفصل السادس بالتقرير للتفاصيل الكاملة.",
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    print(f"تم تسجيل الإصدار '{VERSION_NAME}' بنجاح - id = {new_id}")
    conn.close()


if __name__ == "__main__":
    register()
