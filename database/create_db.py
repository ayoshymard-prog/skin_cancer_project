"""
create_db.py (v2 - موسّع)
----------------------------
إنشاء قاعدة بيانات موسّعة لمشروع تشخيص سرطان الجلد المبكر.

الجداول:
- users              : الأطباء / المستخدمون
- patients           : المرضى (بيانات أوسع)
- lesion_types       : مرجع فئات الآفات الجلدية السبعة (حسب HAM10000)
- images             : الصور المرفوعة
- diagnosis_results  : نتائج تحليل النموذج الذكي
- appointments       : مواعيد المتابعة
- treatments         : خطط العلاج المرتبطة بالتشخيص
- model_versions     : سجل إصدارات النموذج الذكي ودقتها
- audit_log          : سجل تدقيق لعمليات المستخدمين
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "skin_cancer.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'doctor',
        specialization TEXT,
        phone TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT CHECK(gender IN ('male', 'female')),
        skin_type TEXT,
        family_history INTEGER DEFAULT 0,
        phone TEXT,
        address TEXT,
        national_id TEXT,
        medical_notes TEXT,
        registered_by INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (registered_by) REFERENCES users(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lesion_types (
        code TEXT PRIMARY KEY,
        name_ar TEXT NOT NULL,
        name_en TEXT NOT NULL,
        is_malignant INTEGER NOT NULL,
        typical_share_percent REAL,
        description TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        body_location TEXT,
        image_source TEXT DEFAULT 'clinic_camera',
        upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
        uploaded_by INTEGER,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
        FOREIGN KEY (uploaded_by) REFERENCES users(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_name TEXT UNIQUE NOT NULL,
        architecture TEXT,
        training_dataset TEXT,
        accuracy REAL,
        precision_score REAL,
        recall_score REAL,
        f1_score REAL,
        trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diagnosis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        lesion_type_code TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        risk_level TEXT,
        model_version_id INTEGER,
        reviewed_by_doctor INTEGER DEFAULT 0,
        doctor_notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
        FOREIGN KEY (lesion_type_code) REFERENCES lesion_types(code),
        FOREIGN KEY (model_version_id) REFERENCES model_versions(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        appointment_date TEXT NOT NULL,
        reason TEXT,
        status TEXT DEFAULT 'scheduled',
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
        FOREIGN KEY (doctor_id) REFERENCES users(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS treatments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        diagnosis_id INTEGER NOT NULL,
        treatment_type TEXT,
        start_date TEXT,
        status TEXT DEFAULT 'ongoing',
        outcome_notes TEXT,
        FOREIGN KEY (diagnosis_id) REFERENCES diagnosis_results(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        target_table TEXT,
        target_id INTEGER,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()
    print(f"تم إنشاء قاعدة البيانات الموسّعة بنجاح في: {DB_PATH}")


if __name__ == "__main__":
    create_database()
