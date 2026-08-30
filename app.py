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
sys.path.append(os.path.join(os.path.dirname(__file__), "database"))
from importlib import import_module
predict_module = import_module("04_predict")
create_db_module = import_module("create_db")
seed_data_module = import_module("seed_data")
register_model_module = import_module("register_model_version")

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "skin_cancer.db")

# st.set_page_config يجب أن يكون أول أمر Streamlit يُنفَّذ بالسكربت بالكامل
st.set_page_config(page_title="تشخيص سرطان الجلد المبكر", layout="centered")


@st.cache_resource
def ensure_db_ready():
    """
    تتأكد من وجود الجداول وتنشئها تلقائياً إذا لم تكن موجودة (مهم لبيئات
    سحابية مثل Streamlit Cloud حيث لا يُرفَع ملف قاعدة البيانات نفسه على
    Git). @st.cache_resource يضمن تنفيذها مرة واحدة فقط طوال عمر الجلسة،
    وليس مع كل إعادة تشغيل للسكربت.
    """
    create_db_module.create_database()

    conn = sqlite3.connect(DB_PATH)
    patients_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()

    if patients_count == 0:
        seed_data_module.seed()

    register_model_module.register()
    return True


ensure_db_ready()

# ================== نظام التصميم ==================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #0B2740;
        --teal: #028090;
        --seafoam: #00A896;
        --mint: #02C39A;
        --amber: #C98A2E;
        --clay: #B23A32;
        --bg: #F5F9F8;
        --card: #FFFFFF;
        --muted: #5B7280;
        --border: #E1E8E6;
    }

    html, body, .stApp {
        direction: rtl;
        text-align: right;
        background: var(--bg);
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        color: var(--ink);
    }
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
        font-weight: 700;
        color: var(--ink);
    }
    code, .stCaption, [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* شريط العنوان */
    .clinic-header {
        background: linear-gradient(135deg, var(--ink) 0%, #123A5C 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 4px 18px rgba(11,39,64,0.18);
    }
    .clinic-header h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        margin: 0 0 6px 0;
    }
    .clinic-header p {
        color: #BFE3DE;
        margin: 0;
        font-size: 14px;
    }
    .clinic-header .eyebrow {
        display: inline-block;
        background: var(--teal);
        color: white;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }

    /* تبويبات على شكل كبسولات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--card);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--ink) !important;
        color: white !important;
    }

    /* الأزرار */
    .stButton > button {
        background: var(--ink);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: background 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--teal);
        color: white;
    }

    /* بطاقات المؤشرات */
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }

    /* منطقة رفع الملفات */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 1.5px dashed var(--teal) !important;
        background: #F0F8F7;
    }

    /* تنبيهات */
    [data-testid="stAlert"] { border-radius: 12px; }

    /* بطاقة نتيجة التشخيص المخصصة */
    .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-top: 12px;
        box-shadow: 0 2px 12px rgba(11,39,64,0.06);
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_HEX = {"low": "#02C39A", "medium": "#C98A2E", "high": "#B23A32"}
RISK_LABEL_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "مرتفعة"}
CLASS_NAMES_AR = {
    "akiec": "توسّف شعاعي (قبل سرطاني)",
    "bcc":   "سرطان الخلايا القاعدية",
    "bkl":   "توسّف حميد",
    "df":    "ورم ليفي جلدي (حميد)",
    "mel":   "الميلانوما",
    "nv":    "شامة عادية (حميدة)",
    "vasc":  "آفة وعائية",
}


def render_gauge_card(confidence: float, risk: str, class_name_ar: str, code: str) -> str:
    """يبني بطاقة نتيجة بقرص قياس دائري (SVG) بلون يعكس مستوى الخطورة."""
    pct = confidence * 100
    color = RISK_HEX[risk]
    circumference = 2 * 3.14159 * 54
    offset = circumference * (1 - confidence)
    return f"""
    <div class="result-card">
      <div style="display:flex; align-items:center; gap:28px; flex-wrap:wrap; justify-content:center;">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <circle cx="70" cy="70" r="54" fill="none" stroke="#E1E8E6" stroke-width="12"/>
          <circle cx="70" cy="70" r="54" fill="none" stroke="{color}" stroke-width="12"
                  stroke-linecap="round" stroke-dasharray="{circumference:.1f}"
                  stroke-dashoffset="{offset:.1f}"
                  transform="rotate(-90 70 70)"/>
          <text x="70" y="65" text-anchor="middle" font-family="IBM Plex Mono, monospace"
                font-size="24" font-weight="600" fill="{color}">{pct:.1f}%</text>
          <text x="70" y="85" text-anchor="middle" font-family="IBM Plex Mono, monospace"
                font-size="10" fill="#5B7280">CONFIDENCE</text>
        </svg>
        <div style="text-align:center; min-width:180px;">
          <div style="font-size:20px; font-weight:700; color:var(--ink); margin-bottom:4px;">{class_name_ar}</div>
          <div style="font-family:'IBM Plex Mono',monospace; color:#5B7280; font-size:13px; margin-bottom:12px;">{code}</div>
          <span class="risk-badge" style="background:{color};">مستوى الخطورة: {RISK_LABEL_AR[risk]}</span>
        </div>
      </div>
    </div>
    """


st.markdown(
    """
    <div class="clinic-header">
        <span class="eyebrow">مشروع تخرج — ذكاء صنعي</span>
        <h1>🩺 برنامج تشخيص مرض سرطان الجلد بشكل مبكر</h1>
        <p>سلمى محمد عيسى · إشراف د. محمد حجوز</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

                        st.markdown(
                            render_gauge_card(
                                result["confidence_score"], risk, CLASS_NAMES_AR[cls], cls
                            ),
                            unsafe_allow_html=True,
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
