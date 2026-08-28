"""
01_data_exploration.py
------------------------
الهدف: استكشاف قاعدة بيانات HAM10000 (صور سرطان الجلد)
يعمل هذا السكربت داخل Google Colab بعد تحميل البيانات من Kaggle أو ISIC Archive.

خطوات التحميل من Colab:
1) رفع ملف kaggle.json (API Token من حسابك على Kaggle)
2) تنفيذ:
   !pip install kaggle
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   !unzip -q skin-cancer-mnist-ham10000.zip -d ham10000_data
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# مسار الميتاداتا بعد فك الضغط (عدّليه بحسب مكان التحميل الفعلي في Colab)
METADATA_PATH = "ham10000_data/HAM10000_metadata.csv"
IMAGES_DIR = "ham10000_data/HAM10000_images"

# قاموس شرح تصنيفات المرض (حسب توصيف HAM10000 الرسمي)
DIAGNOSIS_LABELS = {
    "akiec": "Actinic keratoses (توسف شعاعي / مرحلة أولى محتملة للسرطان)",
    "bcc":   "Basal cell carcinoma (سرطان الخلايا القاعدية)",
    "bkl":   "Benign keratosis (توسف حميد)",
    "df":    "Dermatofibroma (ورم ليفي جلدي - حميد)",
    "mel":   "Melanoma (الميلانوما - الأخطر)",
    "nv":    "Melanocytic nevi (شامة عادية - حميدة)",
    "vasc":  "Vascular lesions (آفات وعائية)",
}


def load_metadata(path=METADATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"لم يتم العثور على {path}. تأكدي من تحميل وفك ضغط البيانات أولاً."
        )
    df = pd.read_csv(path)
    return df


def explore(df: pd.DataFrame):
    print("عدد الصور الإجمالي:", len(df))
    print("\nتوزيع الفئات (dx):")
    print(df["dx"].value_counts())

    print("\nنسبة كل فئة من الإجمالي:")
    print((df["dx"].value_counts(normalize=True) * 100).round(2))

    # رسم توزيع الفئات
    plt.figure(figsize=(8, 5))
    df["dx"].value_counts().plot(kind="bar")
    plt.title("توزيع فئات الآفات الجلدية في HAM10000")
    plt.xlabel("الفئة")
    plt.ylabel("عدد الصور")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("\nتم حفظ الرسم البياني: class_distribution.png")

    # فحص القيم المفقودة
    print("\nالقيم المفقودة في كل عمود:")
    print(df.isnull().sum())


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    تبسيط المشكلة إلى تصنيف ثنائي (سليم/مشتبه بالخطورة) كخطوة أولى قبل
    الانتقال للتصنيف متعدد الفئات.
    mel و bcc و akiec تعتبر أكثر خطورة (malignant/pre-malignant)
    """
    malignant_classes = {"mel", "bcc", "akiec"}
    df["binary_label"] = df["dx"].apply(
        lambda x: "malignant_suspect" if x in malignant_classes else "benign"
    )
    return df


if __name__ == "__main__":
    df = load_metadata()
    explore(df)
    df = add_binary_label(df)
    print("\nتوزيع التصنيف الثنائي المبسّط:")
    print(df["binary_label"].value_counts())
    df.to_csv("metadata_with_binary_label.csv", index=False)
    print("\nتم حفظ الملف الناتج: metadata_with_binary_label.csv")
