import streamlit as st
from get_audio import get_audio
from get_ppt import get_ppt
from pdf_conversion import convert_to_pdf
from get_avatar import get_avatar
from get_overlay import get_fin_video
st.set_page_config(page_title="PowerPoint Updater", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Get PPTx","PDF Conversion","Get Audio","Avatar Generation","Overlay Videos"])

# Show the selected page
if page == "Get PPTx":
    get_ppt()
elif page == "Get Audio":
    get_audio()
elif page == "PDF Conversion":
    convert_to_pdf()
elif page == "Avatar Generation":
    get_avatar()
elif page == "Overlay Videos":
    get_fin_video()
