1. clone the repo
    git clone https://github.com/KarthikAvinash/demo.git

2. Create a virtual environment using the followin command:
    python -m venv kar

3. Activate the virtual environment using the followin command:
    kar\Scripts\activate

4. Install all the dependencies using the requirements.txt file using the following code:
    pip install -r requirements.txt

5. Set all the credentials in the .env file in this directory.

6. Run the main app that has all the navigation to various steps to convert a ppt to a video in order:
    streamlit run app.py

Note: You need to have a template pptx file that has placeholders - currently these are the allowed placeholders:
- "title" for title in a give slide, 
- "stitle" sub title,
- "txt" normal text,
- "img" image - also the size of the text box will be the size of the image. (currently the images are from stock images, will convert this into google image links soon)

- "points" *- will format the points as bullet points in a text box, but it is not yet implemented.

Make sure that there is a stable interner connection when running this application.
