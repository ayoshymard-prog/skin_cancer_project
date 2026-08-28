"""
seed_data.py
--------------
توليد بيانات اصطناعية (Synthetic) واسعة لقاعدة بيانات المشروع، لأغراض
العرض والاختبار فقط — هذه ليست بيانات مرضى حقيقيين.

مصدر النسب المستخدمة لتوزيع فئات الآفات: الورقة العلمية الأصلية لمجموعة
بيانات HAM10000 (Tschandl, Rosendahl & Kittler, 2018) التي نشرت التوزيع
الحقيقي للفئات السبع على 10,015 صورة:
    nv: 6705 (67.0%) | mel: 1113 (11.1%) | bkl: 1099 (11.0%)
    bcc: 514 (5.1%)  | akiec: 327 (3.3%) | vasc: 142 (1.4%) | df: 115 (1.1%)

استخدمنا هذه النسب فقط لتوليد بيانات تحاكي واقع عدم توازن الفئات،
دون استخدام أي صور أو سجلات مرضى فعلية.
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "skin_cancer.db")
random.seed(42)

# ---------------------------------------------------------------
# 1) بيانات مرجعية: فئات الآفات السبع (النسب من الورقة العلمية الأصلية)
# ---------------------------------------------------------------
LESION_TYPES = [
    ("akiec", "توسّف شعاعي", "Actinic keratoses", 1, 3.3,
     "آفة قبل سرطانية ناتجة عن التعرض الشمسي المزمن"),
    ("bcc", "سرطان الخلايا القاعدية", "Basal cell carcinoma", 1, 5.1,
     "أكثر أنواع سرطان الجلد شيوعاً، بطيء الانتشار"),
    ("bkl", "توسّف حميد", "Benign keratosis-like lesions", 0, 11.0,
     "آفة حميدة شائعة لدى كبار السن"),
    ("df", "ورم ليفي جلدي", "Dermatofibroma", 0, 1.1,
     "ورم حميد صغير وصلب تحت الجلد"),
    ("mel", "الميلانوما", "Melanoma", 1, 11.1,
     "أخطر أنواع سرطان الجلد وأسرعها انتشاراً"),
    ("nv", "شامة عادية", "Melanocytic nevi", 0, 67.0,
     "أكثر الآفات الجلدية شيوعاً، حميدة في الغالب"),
    ("vasc", "آفة وعائية", "Vascular lesions", 0, 1.4,
     "آفات ناتجة عن تشوهات الأوعية الدموية السطحية"),
]

RISK_BY_MALIGNANT = {1: ["medium", "high"], 0: ["low"]}

FIRST_NAMES_M = ["أحمد", "محمد", "علي", "خالد", "يوسف", "عمر", "سامر", "زياد",
                  "حسن", "طارق", "وائل", "رامي", "فادي", "ماهر", "نبيل"]
FIRST_NAMES_F = ["سلمى", "لينا", "رنا", "هبة", "ريم", "دانا", "نور", "سارة",
                  "ياسمين", "لمى", "غنى", "ديمة", "رزان", "جنى", "ميار"]
LAST_NAMES = ["الأحمد", "الحسن", "الخطيب", "العلي", "السيد", "المصري",
              "الحلبي", "الشامي", "قاسم", "درويش", "زيدان", "عيسى",
              "حجازي", "النجار", "الشريف"]

BODY_LOCATIONS = ["الظهر", "الوجه", "الذراع", "الساق", "الصدر", "فروة الرأس",
                   "الكتف", "الرقبة", "اليد", "القدم"]

DOCTORS = [
    ("د. محمد حجوز", "supervisor@example.com", "جلدية وأورام", "0991000001"),
    ("د. رنا الأتاسي", "rana.atassi@example.com", "طب جلدية", "0991000002"),
    ("د. باسل مراد", "basel.murad@example.com", "أورام جلدية", "0991000003"),
]


def weighted_lesion_sample(n):
    """توليد عيّنة فئات آفات موزّعة حسب النسب الحقيقية لـ HAM10000"""
    codes = [l[0] for l in LESION_TYPES]
    weights = [l[4] for l in LESION_TYPES]
    return random.choices(codes, weights=weights, k=n)


def random_date(days_back=365):
    d = datetime.now() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d %H:%M:%S")


def seed(num_patients=60, images_per_patient_max=3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # -------- lesion_types --------
    cursor.executemany(
        """INSERT OR IGNORE INTO lesion_types
           (code, name_ar, name_en, is_malignant, typical_share_percent, description)
           VALUES (?, ?, ?, ?, ?, ?)""",
        LESION_TYPES,
    )

    # -------- users (doctors) --------
    doctor_ids = []
    for name, email, spec, phone in DOCTORS:
        cursor.execute(
            """INSERT OR IGNORE INTO users
               (full_name, email, password_hash, role, specialization, phone)
               VALUES (?, ?, ?, 'doctor', ?, ?)""",
            (name, email, "hashed_pw_demo", spec, phone),
        )
    cursor.execute("SELECT id FROM users")
    doctor_ids = [r[0] for r in cursor.fetchall()]

    # -------- model_versions --------
    cursor.execute(
        """INSERT OR IGNORE INTO model_versions
           (version_name, architecture, training_dataset, accuracy,
            precision_score, recall_score, f1_score, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("v0.1-baseline", "MobileNetV2 (transfer learning)", "HAM10000",
         None, None, None, None,
         "نموذج أولي لم يُدرَّب بعد على البيانات الفعلية - قيد الإعداد"),
    )
    cursor.execute("SELECT id FROM model_versions LIMIT 1")
    model_version_id = cursor.fetchone()[0]

    # -------- patients + images + diagnosis_results --------
    for _ in range(num_patients):
        gender = random.choice(["male", "female"])
        first = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        age = random.randint(18, 85)
        skin_type = random.choice(["I", "II", "III", "IV", "V", "VI"])
        family_history = random.choices([0, 1], weights=[80, 20])[0]
        doctor = random.choice(doctor_ids)

        cursor.execute(
            """INSERT INTO patients
               (full_name, age, gender, skin_type, family_history, phone,
                address, national_id, registered_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"{first} {last}", age, gender, skin_type, family_history,
                f"09{random.randint(10000000,99999999)}",
                random.choice(["دمشق", "حلب", "حمص", "اللاذقية", "طرطوس", "درعا"]),
                str(random.randint(1000000000, 9999999999)),
                doctor, random_date(400),
            ),
        )
        patient_id = cursor.lastrowid

        num_images = random.randint(1, images_per_patient_max)
        lesion_codes = weighted_lesion_sample(num_images)

        for i, code in enumerate(lesion_codes):
            is_malignant = next(l[3] for l in LESION_TYPES if l[0] == code)
            upload_date = random_date(300)

            cursor.execute(
                """INSERT INTO images
                   (patient_id, image_path, body_location, image_source,
                    upload_date, uploaded_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    f"/data/images/patient_{patient_id}_img_{i+1}.jpg",
                    random.choice(BODY_LOCATIONS),
                    random.choice(["dermatoscope", "clinic_camera", "mobile"]),
                    upload_date, doctor,
                ),
            )
            image_id = cursor.lastrowid

            confidence = round(random.uniform(0.62, 0.98), 3)
            risk_level = random.choice(RISK_BY_MALIGNANT[is_malignant])

            cursor.execute(
                """INSERT INTO diagnosis_results
                   (image_id, lesion_type_code, confidence_score, risk_level,
                    model_version_id, reviewed_by_doctor, doctor_notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id, code, confidence, risk_level, model_version_id,
                    random.choices([0, 1], weights=[30, 70])[0],
                    "بيانات تجريبية - بانتظار مراجعة طبية فعلية"
                    if risk_level != "low" else None,
                    upload_date,
                ),
            )
            diagnosis_id = cursor.lastrowid

            # مواعيد متابعة للحالات متوسطة/عالية الخطورة فقط
            if risk_level in ("medium", "high"):
                cursor.execute(
                    """INSERT INTO appointments
                       (patient_id, doctor_id, appointment_date, reason, status)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        patient_id, doctor,
                        (datetime.now() + timedelta(days=random.randint(3, 30)))
                        .strftime("%Y-%m-%d"),
                        "متابعة نتيجة تحليل مشتبه بها",
                        random.choice(["scheduled", "completed"]),
                    ),
                )

            # خطة علاج للحالات عالية الخطورة فقط
            if risk_level == "high":
                cursor.execute(
                    """INSERT INTO treatments
                       (diagnosis_id, treatment_type, start_date, status, outcome_notes)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        diagnosis_id,
                        random.choice(["خزعة تشخيصية", "استئصال جراحي", "مراقبة دورية"]),
                        upload_date[:10],
                        random.choice(["ongoing", "completed"]),
                        "بيانات تجريبية لأغراض العرض",
                    ),
                )

        # سجل تدقيق لعملية تسجيل المريض
        cursor.execute(
            """INSERT INTO audit_log (user_id, action, target_table, target_id, timestamp)
               VALUES (?, 'register_patient', 'patients', ?, ?)""",
            (doctor, patient_id, random_date(400)),
        )

    conn.commit()

    # -------- طباعة ملخص --------
    cursor.execute("SELECT COUNT(*) FROM patients")
    print("عدد المرضى:", cursor.fetchone()[0])
    cursor.execute("SELECT COUNT(*) FROM images")
    print("عدد الصور:", cursor.fetchone()[0])
    cursor.execute("SELECT COUNT(*) FROM diagnosis_results")
    print("عدد نتائج التحليل:", cursor.fetchone()[0])
    cursor.execute("""
        SELECT lesion_type_code, COUNT(*)
        FROM diagnosis_results GROUP BY lesion_type_code ORDER BY COUNT(*) DESC
    """)
    print("\nتوزيع نتائج التحليل حسب الفئة (يحاكي نسب HAM10000):")
    for code, count in cursor.fetchall():
        print(f"  {code}: {count}")

    conn.close()


if __name__ == "__main__":
    seed()
