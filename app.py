import streamlit as st
from utils.load_model import load_model
from utils.preprocessing import transform
from utils.prediction import predict
from utils.leaf_crop import grabcut_leaf
from PIL import Image

# ======================
# JUDUL
# ======================
st.title("🌿 KLASIFIKASI PENYAKIT DAUN CABAI")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ======================
# DESKRIPSI PENYAKIT
# ======================
deskripsi = {
    "Sehat": "Daun dalam kondisi normal tanpa indikasi penyakit.",
    "Bercak": "Daun menunjukkan adanya bercak atau noda yang biasanya disebabkan oleh infeksi jamur atau bakteri.",
    "Keriting": "Daun mengalami deformasi (keriting) yang biasanya disebabkan oleh hama atau infeksi.",
    "Kuning": "Daun mengalami perubahan warna menjadi kuning yang dapat disebabkan oleh kekurangan nutrisi atau penyakit."
}

# ======================
# UPLOAD FILE
# ======================
uploaded_rgb = st.file_uploader(
    "Upload RGB",
    type=["jpg", "png", "jpeg"]
)

uploaded_th = st.file_uploader(
    "Upload Thermal",
    type=["jpg", "png", "jpeg"]
)

# ======================
# VALIDASI
# ======================
if uploaded_rgb and uploaded_th:

    if st.button("Prediksi"):

        # ======================
        # LOAD IMAGE
        # ======================
        rgb = Image.open(uploaded_rgb).convert("RGB")
        th = Image.open(uploaded_th).convert("RGB")

        rgb = rgb.resize((512, 512))
        th = th.resize((512, 512))

        # ======================
        # RGB PREPROCESS
        # ======================
        # rgb = grabcut_leaf(rgb)

        # ======================
        # TENSOR
        # ======================
        rgb_tensor = transform(rgb).unsqueeze(0)
        th_tensor = transform(th).unsqueeze(0)

        # ======================
        # PREDIKSI
        # ======================
        label, conf, probs = predict(
            rgb_tensor,
            th_tensor,
            model
        )

        # ======================
        # GAMBAR
        # ======================
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                rgb,
                caption="RGB",
                use_container_width=True
            )

        with col2:
            st.image(
                th,
                caption="Thermal",
                use_container_width=True
            )

        # ======================
        # HASIL
        # ======================
        st.markdown("## 📊 Hasil Analisis")

        st.success(
            f"🌿 Klasifikasi: {label}"
        )

        st.write(
            f"📈 Confidence: {conf*100:.2f}%"
        )

        # ======================
        # PROBABILITAS
        # ======================
        st.markdown(
            "### 📊 Probabilitas Tiap Kelas"
        )

        classes = [
            "Bercak",
            "Keriting",
            "Kuning",
            "Sehat"
        ]

        for cls, p in zip(classes, probs):

            st.progress(float(p))

            st.write(
                f"{cls}: {float(p)*100:.2f}%"
            )

        # ======================
        # DESKRIPSI
        # ======================
        st.markdown(
            "### 🩺 Deskripsi"
        )

        st.info(
            deskripsi.get(
                label,
                "Deskripsi tidak tersedia"
            )
        )