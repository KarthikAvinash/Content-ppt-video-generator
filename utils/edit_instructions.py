import streamlit as st
import os
from utils.get_ppt import *

ppt_dir = "ppts"
ppt_path = os.path.join(ppt_dir, "template.pptx")


# File to store the instructions
INSTRUCTIONS_FILE = "instructions.txt"

def load_instructions():
    """Load instructions from the text file."""
    if os.path.exists(INSTRUCTIONS_FILE):
        with open(INSTRUCTIONS_FILE, "r") as f:
            return f.read().splitlines()
    else:
        # Default sample data if the file doesn't exist
        return [
            "Text 1 for Slide 1", "Text 2 for Slide 1", "Text 3 for Slide 1",
            "Text 1 for Slide 2", "Text 2 for Slide 2", "Text 3 for Slide 2",
            "Text 4 for Slide 2", "Text 5 for Slide 2", "Text 6 for Slide 2",
            "Text 7 for Slide 2", "Text 8 for Slide 2", "Text 9 for Slide 2",
            "Text 10 for Slide 2", "Text 11 for Slide 2",
            "Text 1 for Slide 3", "Text 2 for Slide 3", "Text 3 for Slide 3",
            "Text 4 for Slide 3", "Text 5 for Slide 3", "Text 6 for Slide 3",
            "Text 7 for Slide 3", "Text 8 for Slide 3", "Text 9 for Slide 3"
        ]

def save_instructions(instructions):
    """Save instructions to the text file."""
    with open(INSTRUCTIONS_FILE, "w") as f:
        f.write("\n".join(instructions))

def edit_instructions():
    # Load instructions from file
    instructions = load_instructions()

    instruction_count = {'1': 3, '2': 11, '3': 9}

    # Initialize session state to store edited text
    if 'edited_text' not in st.session_state:
        st.session_state.edited_text = instructions.copy()

    st.title("Text Editor for Multiple Slides")

    # Create tabs for each slide
    tabs = st.tabs([f"Slide {i}" for i in instruction_count.keys()])

    # Track the current index in the instructions list
    current_index = 0

    # Loop through each slide
    for i, (slide, count) in enumerate(instruction_count.items()):
        with tabs[i]:
            st.header(f"Slide {slide}")
            
            # Create text areas for each text in the slide
            for j in range(count):
                text_index = current_index + j
                new_text = st.text_area(
                    f"Text {j+1}",
                    st.session_state.edited_text[text_index],
                    key=f"text_{slide}_{j}"
                )

                # If the text is edited, update session state and write it to the file immediately
                if new_text != st.session_state.edited_text[text_index]:
                    st.session_state.edited_text[text_index] = new_text
                    save_instructions(st.session_state.edited_text)  # Save to the file immediately
            
            current_index += count

    # Create a button to reload the instructions from the file
    if st.button("Reload Instructions"):
        instructions = load_instructions()
        st.session_state.edited_text = instructions.copy()
        st.success("Instructions reloaded from file!")

    # Display the updated original list
    st.subheader("Updated Original List (Stored in File)")
    st.write(st.session_state.edited_text)

    if st.button("Update the ppt with new text!"):
            update_presentation_with_instructions(ppt_path, instructions, 'updated_presentation.pptx')
            instructions = remove_image_placeholder(instructions)

            output_path = 'updated_presentation.pptx'
            access_key = "IOuG8tW-MhjIxRKKXrhd079mIqGArt9iqF4NLn1AFsE"
            replace_placeholders_with_text(ppt_path, output_path, instructions, access_key)
            
            # Provide download option for updated presentation
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Updated Presentation",
                    data=file,
                    file_name="updated_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
