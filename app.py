"""
app.py
--------
واجهة استخدام تجريبية (GUI) للأطباء - برنامج تشخيص سرطان الجلد المبكر
تشغيل: streamlit run app.py
"""

import streamlit as st
import sqlite3
import os
import sys
import tempfile
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from importlib import import_module
predict_module = import_module("04_predict")

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "skin_cancer.db")

# أسماء عرض عربية لفئات HAM10000 السبع
CLASS_NAMES_AR = {
    "akiec": "توسّف شعاعي (قبل سرطاني)",
    "bcc":   "سرطان الخلايا القاعدية",
    "bkl":   "توسّف حميد",
    "df":    "ورم ليفي جلدي (حميد)",
    "mel":   "الميلانوما",
    "nv":    "شامة عادية (حميدة)",
    "vasc":  "آفة وعائية",
}
RISK_LABEL_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "مرتفعة"}
RISK_COLOR = {"low": "success", "medium": "warning", "high": "error"}

st.set_page_config(page_title="تشخيص سرطان الجلد المبكر", layout="centered")

# دعم واجهة من اليمين لليسار (RTL) للعربية
st.markdown(
    """
    <style>
    body, .stApp { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 برنامج تشخيص مرض سرطان الجلد بشكل مبكر")
st.caption("مشروع تخرج - سلمى محمد عيسى | إشراف: د. محمد حجوز")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["تشخيص صورة جديدة", "سجل المرضى", "لوحة إحصائيات", "عن المشروع"]
)

# ------------------- التبويب 1: تشخيص صورة -------------------
with tab1:
    st.subheader("رفع صورة جلدية للتحليل")

    # اختيار مريض موجود أو تسجيل مريض جديد
    conn = sqlite3.connect(DB_PATH)
    patients_df_rows = conn.execute(
        "SELECT id, full_name FROM patients ORDER BY full_name"
    ).fetchall()
    conn.close()

    patient_mode = st.radio("المريض", ["اختيار مريض موجود", "تسجيل مريض جديد"], horizontal=True)

    patient_id = None
    if patient_mode == "اختيار مريض موجود" and patients_df_rows:
        options = {f"{name} (#{pid})": pid for pid, name in patients_df_rows}
        chosen = st.selectbox("اختاري المريض", list(options.keys()))
        patient_id = options[chosen]
    else:
        patient_name = st.text_input("اسم المريض")
        age = st.number_input("العمر", min_value=0, max_value=120, value=30)
        gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
        if st.button("حفظ بيانات المريض") and patient_name:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO patients (full_name, age, gender) VALUES (?, ?, ?)",
                (patient_name, age, "male" if gender == "ذكر" else "female"),
            )
            patient_id = cur.lastrowid
            conn.commit()
            conn.close()
            st.success(f"تم تسجيل المريض بنجاح (#{patient_id})")

    body_location = st.text_input("موقع الآفة على الجسم (مثال: الظهر، الذراع)")

    uploaded_file = st.file_uploader(
        "اختر صورة الآفة الجلدية", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="الصورة المرفوعة", width=300)

        if st.button("تشغيل التحليل", type="primary"):
            if patient_id is None:
                st.error("الرجاء اختيار مريض أو تسجيل مريض جديد أولاً.")
            else:
                with st.spinner("جاري تحليل الصورة بواسطة النموذج الذكي..."):
                    # حفظ الصورة المرفوعة مؤقتاً على القرص للمعالجة
                    tmp_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
                    os.makedirs(tmp_dir, exist_ok=True)
                    image_filename = f"patient_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    image_path = os.path.join(tmp_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        # 1) تشغيل النموذج الحقيقي
                        result = predict_module.predict_image(image_path)

                        # 2) حفظ سجل الصورة بقاعدة البيانات
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute(
                            """INSERT INTO images (patient_id, image_path, body_location, image_source)
                               VALUES (?, ?, ?, 'clinic_camera')""",
                            (patient_id, image_path, body_location),
                        )
                        image_id = cur.lastrowid
                        conn.commit()
                        conn.close()

                        # 3) حفظ نتيجة التحليل مرتبطة بإصدار النموذج المسجَّل
                        model_version_id = predict_module.get_active_model_version_id()
                        diagnosis_id = predict_module.save_result_to_db(
                            image_id, result, model_version_id
                        )

                        # 4) عرض النتيجة للطبيب
                        cls = result["predicted_class"]
                        risk = result["risk_level"]
                        st.divider()
                        st.subheader("نتيجة التحليل")
                        col1, col2 = st.columns(2)
                        col1.metric("التصنيف", CLASS_NAMES_AR[cls])
                        col2.metric("نسبة الثقة", f"{result['confidence_score']*100:.1f}%")

                        getattr(st, RISK_COLOR[risk])(
                            f"مستوى الخطورة: **{RISK_LABEL_AR[risk]}**"
                        )

                        with st.expander("عرض احتمالات كل الفئات"):
                            for code, prob in sorted(
                                result["all_probabilities"].items(),
                                key=lambda x: x[1], reverse=True
                            ):
                                st.write(f"{CLASS_NAMES_AR[code]} ({code}): {prob*100:.1f}%")

                        if risk in ("medium", "high"):
                            st.info(
                                "⚠️ تم جدولة موعد متابعة تلقائياً لهذه الحالة "
                                "نظراً لمستوى الخطورة."
                            )

                        st.caption(
                            f"تم حفظ النتيجة بقاعدة البيانات (رقم السجل: {diagnosis_id}). "
                            "يمكن للطبيب مراجعتها وإضافة ملاحظاته من تبويب سجل المرضى."
                        )

                    except FileNotFoundError as e:
                        st.error(
                            "⚠️ لم يتم العثور على ملف النموذج المدرَّب. "
                            "تأكدي من وجود best_model_efficientnet.h5 بمجلد models/."
                        )
                        st.exception(e)


# ------------------- التبويب 2: سجل المرضى -------------------
with tab2:
    st.subheader("سجل المرضى المسجلين في النظام")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, full_name, age, gender, address, family_history, created_at
               FROM patients ORDER BY created_at DESC"""
        )
        rows = cursor.fetchall()
        conn.close()

        if rows:
            st.caption(f"إجمالي عدد المرضى المسجلين: {len(rows)}")
            st.dataframe(
                [
                    {
                        "الرقم": r[0],
                        "الاسم": r[1],
                        "العمر": r[2],
                        "الجنس": "ذكر" if r[3] == "male" else "أنثى",
                        "المحافظة": r[4],
                        "تاريخ عائلي": "نعم" if r[5] else "لا",
                        "تاريخ التسجيل": r[6],
                    }
                    for r in rows
                ],
                use_container_width=True,
            )
        else:
            st.info("لا يوجد مرضى مسجلون بعد.")
    except sqlite3.OperationalError:
        st.error(
            "قاعدة البيانات غير موجودة أو غير محدّثة. شغّلي أولاً: "
            "python database/create_db.py ثم python database/seed_data.py"
        )

# ------------------- التبويب 3: لوحة إحصائيات -------------------
with tab3:
    st.subheader("توزيع نتائج التحليل حسب فئة الآفة")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT lt.name_ar, lt.is_malignant, COUNT(*) as cnt
               FROM diagnosis_results dr
               JOIN lesion_types lt ON dr.lesion_type_code = lt.code
               GROUP BY lt.code ORDER BY cnt DESC"""
        )
        rows = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM patients")
        total_patients = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM diagnosis_results WHERE risk_level = 'high'"
        )
        high_risk = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM appointments WHERE status='scheduled'")
        pending_appts = cursor.fetchone()[0]
        conn.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("عدد المرضى", total_patients)
        col2.metric("عدد الصور المحلّلة", total_images)
        col3.metric("حالات عالية الخطورة", high_risk)
        col4.metric("مواعيد قادمة", pending_appts)

        if rows:
            st.bar_chart({r[0]: r[2] for r in rows})
            st.dataframe(
                [
                    {
                        "الفئة": r[0],
                        "خطيرة؟": "نعم" if r[1] else "لا",
                        "عدد الحالات": r[2],
                    }
                    for r in rows
                ],
                use_container_width=True,
            )
        else:
            st.info("لا توجد نتائج تحليل بعد.")
    except sqlite3.OperationalError:
        st.error("قاعدة البيانات غير محدّثة. شغّلي create_db.py و seed_data.py أولاً.")

# ------------------- التبويب 4: عن المشروع -------------------
with tab4:
    st.subheader("عن المشروع")
    st.markdown(
        """
    **الهدف:** التشخيص المبكر لسرطان الجلد عن طريق مقارنة صور الخلايا
    السليمة والمصابة باستخدام تقنيات الذكاء الصنعي ورؤية الحاسوب.

    **التقنيات المستخدمة:**
    - Python / TensorFlow / OpenCV
    - نموذج CNN مع Transfer Learning (MobileNetV2)
    - قاعدة بيانات SQLite
    - واجهة استخدام Streamlit

    **حالة المشروع الحالية:** بنية النظام وقاعدة البيانات جاهزة،
    وجاري تدريب النموذج على بيانات HAM10000.
    """
    )
