import streamlit as st
from PIL import Image
import io

from src.config import Config
from src.image_processor import CorporatePortraitProcessor, ImageProcessingError

# --- Configuração da Página ---
st.set_page_config(
    page_title="AI Corporate Headshot",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("👔 AI Corporate Headshot Creator")
    st.markdown("Transforme sua foto em um retrato profissional preservando **100% da sua identidade**.")

    # --- Sidebar: Controles e Upload ---
    st.sidebar.header("1. Upload da Foto")
    uploaded_file = st.sidebar.file_uploader(
        "Escolha uma imagem clara e de frente", 
        type=Config.ALLOWED_TYPES
    )

    st.sidebar.header("2. Ajustes e Estilo")
    bg_style_name = st.sidebar.selectbox("Estilo do Fundo", list(Config.BACKGROUND_STYLES.keys()))
    bg_color = Config.BACKGROUND_STYLES[bg_style_name]
    
    brightness = st.sidebar.slider("Brilho", 0.5, 1.5, 1.0, 0.05)
    contrast = st.sidebar.slider("Contraste", 0.5, 1.5, 1.0, 0.05)
    
    # --- Área Principal: Processamento ---
    if uploaded_file is not None:
        # Validação de Tamanho
        if uploaded_file.size > Config.MAX_FILE_SIZE_BYTES:
            st.error(f"O arquivo excedeu o limite de {Config.MAX_FILE_SIZE_MB}MB. Por favor, envie uma imagem menor.")
            return

        try:
            # Carrega a imagem
            original_image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Antes")
                st.image(original_image, use_column_width=True)

            # Botão de ação
            if st.sidebar.button("Gerar Retrato Profissional", type="primary"):
                with st.spinner("Processando imagem (IA de segmentação em execução)..."):
                    
                    # 1. Validação de Rosto
                    if not CorporatePortraitProcessor.detect_face(original_image):
                        st.warning("⚠️ Nenhum rosto detectado! Certifique-se de usar uma foto de frente e bem iluminada. A qualidade do recorte pode ser afetada.")
                    
                    # 2. Pipeline de Processamento
                    final_image = CorporatePortraitProcessor.create_corporate_portrait(
                        original_image=original_image,
                        bg_color=bg_color,
                        brightness=brightness,
                        contrast=contrast
                    )
                    
                    with col2:
                        st.subheader("Depois (Headshot Corporativo)")
                        st.image(final_image, use_column_width=True)
                        
                        # Preparar imagem para download
                        img_byte_arr = io.BytesIO()
                        final_image.save(img_byte_arr, format='JPEG', quality=95)
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        st.download_button(
                            label="⬇️ Baixar Foto Profissional",
                            data=img_byte_arr,
                            file_name="headshot_corporativo.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                        st.success("Processamento concluído com sucesso!")
                        
        except ImageProcessingError as ipe:
            st.error(f"Erro no processamento: {ipe}")
        except Exception as e:
            st.error("Ocorreu um erro inesperado ao ler a imagem. Verifique se o arquivo não está corrompido.")
    else:
        st.info("👈 Por favor, faça o upload de uma imagem na barra lateral para começar.")

if __name__ == "__main__":
    main()
