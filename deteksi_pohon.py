import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="Deteksi Warna Pohon")
st.title("🌳 Deteksi Warna Pohon & Kalkulator Piksel")

# --- Fungsi Bantuan ---
def load_image_from_uploader(uploader):
    """Membaca file yang diupload menjadi format OpenCV BGR."""
    if uploader is not None:
        try:
            # Membaca gambar dari uploader
            img = Image.open(uploader)
            # Konversi ke format OpenCV (Numpy array)
            img_bgr = np.array(img)
            # Konversi dari RGB (PIL) ke BGR (OpenCV)
            if img_bgr.shape[2] == 4: # Handle RGBA
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2BGR)
            else: # Handle RGB
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            return img_bgr
        except Exception as e:
            st.error(f"Error saat membaca gambar: {e}")
            return None
    return None

# --- Sidebar untuk Pengaturan ---
st.sidebar.header("Panel Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload gambar Gmaps satelit Anda", type=["jpg", "png", "jpeg"])

# Slider untuk rentang warna Hijau di HSV
st.sidebar.subheader("Atur Rentang Warna Hijau (HSV)")
st.sidebar.info("Gunakan slider ini untuk menyesuaikan deteksi warna hijau pada gambar Anda.")

# H (Hue) - Warna
h_min = st.sidebar.slider("H (Min) - Rona Minimum", 0, 179, 35)
h_max = st.sidebar.slider("H (Max) - Rona Maksimum", 0, 179, 85)

# S (Saturation) - Kepekatan
s_min = st.sidebar.slider("S (Min) - Saturasi Minimum", 0, 255, 50)
s_max = st.sidebar.slider("S (Max) - Saturasi Maksimum", 0, 255, 255)

# V (Value) - Kecerahan
v_min = st.sidebar.slider("V (Min) - Kecerahan Minimum", 0, 255, 50)
v_max = st.sidebar.slider("V (Max) - Kecerahan Maksimum", 0, 255, 255)

# --- Panel Utama ---
if uploaded_file is not None:
    img_bgr = load_image_from_uploader(uploaded_file)
    
    if img_bgr is not None:
        # Konversi BGR ke HSV (HSV lebih baik untuk deteksi warna)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Tentukan rentang warna hijau dari slider
        lower_green = np.array([h_min, s_min, v_min])
        upper_green = np.array([h_max, s_max, v_max])
        
        # Buat "Mask" (Topeng) - Inilah inti proses deteksi
        # Piksel dalam rentang hijau akan jadi PUTIH (255), sisanya HITAM (0)
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # (Opsional) Tampilkan hasil deteksi yang sudah diwarnai
        result_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        # Konversi BGR ke RGB untuk ditampilkan
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # Hitung total piksel pohon
        total_tree_pixels = cv2.countNonZero(mask)
        
        # Hitung total piksel gambar
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        
        # Hitung persentase
        percentage = (total_tree_pixels / total_pixels) * 100
        
        # Tampilkan Hasil
        col1, col2 = st.columns(2)
        with col1:
            st.header("Gambar Asli")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)
            
        with col2:
            st.header("Hasil Deteksi")
            st.image(mask, caption="Masker Deteksi (Pohon = Putih)", use_container_width=True)
            # st.image(result_rgb, caption="Area Pohon yang Terdeteksi", use_container_width=True)

        st.divider()
        st.header("📊 Output Total Piksel")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Total Piksel Pohon (Piksel Putih)", value=f"{total_tree_pixels:n}")
        c2.metric(label="Total Piksel Gambar", value=f"{total_pixels:n}")
        c3.metric(label="Persentase Area Pohon", value=f"{percentage:.2f} %")

else:
    st.info("Silakan upload gambar satelit di sidebar untuk memulai deteksi.")
