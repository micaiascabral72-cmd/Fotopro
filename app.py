import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import io
import logging

# --- Configuração Básica ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Classes de Configuração e Erro ---
class Config:
    MAX_FILE_SIZE_MB = 5
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_TYPES = ["jpg", "jpeg", "png"]
    
    BACKGROUND_STYLES = {
        "Estúdio Cinza Escuro (Slate)": (47, 79, 79),
        "Azul Corporativo (Navy)": (0, 0, 128),
        "Cinza Claro (Clean)": (220, 220, 220),
        "Branco Puro (Documento)": (255, 255, 255)
    }

class ImageProcessingError(Exception):
    pass

# --- 2. Lógica de Processamento de Imagem ---
class CorporatePortraitProcessor:
    @staticmethod
    def detect_face(image: Image.Image) -> bool:
        try:
            img_array = np.array(image.convert('L'))
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            return len(faces) > 0
        except Exception as e:
            logger.error(f"Erro na detecção de face: {e}")
            return True # Retorna True como fallback para não travar o fluxo

    @staticmethod
    def adjust_lighting(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
        enhancer_brightness = ImageEnhance.Brightness(image)
        img_bright = enhancer_brightness.enhance(brightness)
        enhancer_contrast = ImageEnhance.Contrast(img_bright)
        return enhancer_contrast.enhance(contrast)

    @staticmethod
    def create_corporate_portrait(original_image: Image.Image, bg_color: tuple, brightness: float, contrast: float) -> Image.Image:
        try:
            adjusted_img = CorporatePortraitProcessor.adjust_lighting(original_image, brightness, contrast)
            foreground = remove(adjusted_img)
            background = Image.new("RGBA", foreground.size, bg_color + (255,))
            background.paste(foreground, (0, 0), foreground)
            return background.convert("RGB")
        except Exception as e:
            raise ImageProcessingError(f"Falha ao processar a imagem: {str(e)}")

# --- 3. Interface do Streamlit ---
st.set_page_config(
    page_title="AI Corporate Headshot",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("👔 AI Corporate Headshot Creator")
    st.markdown("Transforme sua foto em um retrato profissional preservando **100% da sua identidade**.")

    st.sidebar.header("1. Upload da Foto")
    uploaded_file = st.sidebar.file_uploader("Escolha uma imagem clara e de frente", type=Config.ALLOWED_TYPES)

    st.sidebar.header("2. Ajustes e Estilo")
    bg_style_name = st.sidebar.selectbox("Estilo do Fundo", list(Config.BACKGROUND_STYLES.keys()))
    bg_color = Config.BACKGROUND_STYLES[bg_style_name]
    
    brightness = st.sidebar.slider("Brilho", 0.5, 1.5, 1.0, 0.05)
    contrast = st.sidebar.slider("Contraste", 0.5, 1.5, 1.0, 0.05)
    
    if uploaded_file is not None:
        if uploaded_file.size > Config.MAX_FILE_SIZE_BYTES:
            st.error(f"O arquivo excedeu o limite de {Config.MAX_FILE_SIZE_MB}MB.")
            return

        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Antes")
                st.image(original_image, use_column_width=True)

            if st.sidebar.button("Gerar Retrato Profissional", type="primary"):
                with st.spinner("Processando imagem com Inteligência Artificial..."):
                    
                    if not CorporatePortraitProcessor.detect_face(original_image):
                        st.warning("⚠️ Nenhum rosto detectado! A qualidade do recorte pode ser afetada.")
                    
                    final_image = CorporatePortraitProcessor.create_corporate_portrait(
                        original_image=original_image,
                        bg_color=bg_color,
                        brightness=brightness,
                        contrast=contrast
                    )
                    
                    with col2:
                        st.subheader("Depois (Headshot)")
                        st.image(final_image, use_column_width=True)
                        
                        img_byte_arr = io.BytesIO()
                        final_image.save(img_byte_arr, format='JPEG', quality=95)
                        
                        st.download_button(
                            label="⬇️ Baixar Foto Profissional",
                            data=img_byte_arr.getvalue(),
                            file_name="headshot_corporativo.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                        st.success("Processamento concluído com sucesso!")
                        
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
    else:
        st.info("👈 Faça o upload de uma foto na barra lateral para começar.")

if __name__ == "__main__":
    main()
