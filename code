import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io

# --- 1. CÁC HÀM XỬ LÝ  ---
def tien_xu_ly_am_thanh(y, sr):
    stft_matrix = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft_matrix)
    magnitude_log = np.log1p(magnitude)
    max_val = np.max(magnitude_log)
    magnitude_norm = magnitude_log / (max_val + 1e-8)
    return magnitude_norm, phase, magnitude, max_val

def thuc_hien_svd_va_loc(A, k):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_clean = np.zeros_like(S)
    S_clean[:k] = S[:k] 
    A_clean = np.dot(U, np.dot(np.diag(S_clean), VT))
    return A_clean, S

# --- 2. GIAO DIỆN APP ---
st.set_page_config(page_title="SVD Audio Denoiser", layout="wide")
st.title("🎵 Ứng dụng Khử nhiễu Âm thanh bằng Phân tích SVD")
st.markdown("Đề tài: Sử dụng SVD trong miền tần số để loại bỏ nhiễu xe máy.")

# Thanh bên trái (Sidebar)
with st.sidebar:
    st.header("Cấu hình")
    uploaded_file = st.file_uploader("Chọn file âm thanh (wav, mp3)", type=["wav", "mp3"])
    k_value = st.slider("Chọn số giá trị suy biến (k)", min_value=1, max_value=100, value=10)
    process_btn = st.button("🚀 Bắt đầu khử nhiễu")

# Khu vực hiển thị chính
if uploaded_file is not None:
    # Đọc file
    y, sr = librosa.load(uploaded_file, sr=None)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Âm thanh gốc")
        st.audio(uploaded_file)
        
    if process_btn:
        # Thực hiện xử lý
        with st.spinner('Đang phân tích SVD...'):
            A, Pha_goc, mag_goc, max_v = tien_xu_ly_am_thanh(y, sr)
            A_sach_norm, S_values = thuc_hien_svd_va_loc(A, k_value)
            
            # Khôi phục âm thanh
            A_final = np.expm1(A_sach_norm * max_v)
            y_out = librosa.istft(A_final * Pha_goc)
            
            # Xuất file ra bộ nhớ đệm
            buffer = io.BytesIO()
            sf.write(buffer, y_out, sr, format='WAV')
            buffer.seek(0)
            
        with col2:
            st.subheader(f"Âm thanh sau lọc (k={k_value})")
            st.audio(buffer)
            st.download_button("💾 Tải file đã lọc", data=buffer, file_name="output_clean.wav")

        # --- 3. HIỂN THỊ ĐỒ THỊ ---
        st.divider()
        st.subheader("Phân tích trực quan")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Spectrogram sau lọc
        img = librosa.display.specshow(librosa.amplitude_to_db(A_final), sr=sr, x_axis='time', y_axis='hz', ax=ax[0])
        ax[0].set_title(f"Spectrogram sau lọc với k={k_value}")
        fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
        
        # Biểu đồ giá trị suy biến
        ax[1].plot(S_values[:100], 'ro-', markersize=4)
        ax[1].axvline(x=k_value, color='blue', linestyle='--', label=f'Điểm cắt k={k_value}')
        ax[1].set_title("Biểu đồ các giá trị suy biến (Singular Values)")
        ax[1].set_xlabel("Thứ tự giá trị")
        ax[1].set_ylabel("Giá trị")
        ax[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
