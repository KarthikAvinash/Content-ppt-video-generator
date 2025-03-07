# Content to PPT-Video Generator
## Setup Instructions for Windows

1. Clone the repo:
```
git clone https://github.com/KarthikAvinash/Content-ppt-video-generator.git
```

2. Create a virtual environment using the following command:
```
python -m venv kar
```

3. Activate the virtual environment using the following command:
```
kar\Scripts\activate
```

4. Install all the dependencies using the requirements.txt file:
```
pip install -r requirements.txt
```

5. Set all the credentials in the .env file in this directory.
    1. In the root directory of the project, create a new file named .env.
    2. Open the .env file in a text editor.
    3. Add the following lines to the file
    ```
        SPEECH_SERVICE_REGION=southeastasia
        SPEECH_SERVICE_API_KEY=0d5b9a------------------
        GOOGLE_GEMINI_API_KEY=AIza---------------------
    ```
    4. Make sure to replace the placeholders with the right api keys.
    5. Save and close the file.
6. Run the main app that has all the navigation to various steps to convert a ppt to a video in order:
```
streamlit run app.py
```

## Note:

- You need to have a template pptx file that has placeholders - currently these are the allowed placeholders:

    - "title" for title in a given slide
    - "stitle" sub title
    - "txt" normal text
    - "img" image - also the size of the text box will be the size of the image. (currently the images are from stock images, will convert this into google image links soon)
    - "points" *- will format the points as bullet points in a text box, but it is not yet implemented.

- to limit the number of words to be filled for a text box containing a certain type ('txt', 'title', 'img', 'stitle') see the function "def create_prompt(slide_info):" in the file: "utils/ppt_util_functions.py", line - 67.


- Make sure that there is a stable internet connection when running this application.
- Avatar videos generation and overlaying code will be added soon...
