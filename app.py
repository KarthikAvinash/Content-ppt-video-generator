import streamlit as st
from utils.get_audio import get_audio
from utils.get_ppt import get_ppt
from utils.pdf_conversion import convert_to_pdf
from utils.get_avatar import get_avatar
from utils.get_overlay import get_fin_video
from utils.edit_instructions import *
st.set_page_config(page_title="PowerPoint Updater", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Get PPTx", "Edit Instructions", "PDF Conversion","Get Audio","Avatar Generation","Overlay Videos"])

# Show the selected page
if page == "Get PPTx":
    get_ppt()
if page == "Edit Instructions":
    edit_instructions()
elif page == "Get Audio":
    get_audio()
elif page == "PDF Conversion":
    convert_to_pdf()
elif page == "Avatar Generation":
    get_avatar()
elif page == "Overlay Videos":
    get_fin_video()

