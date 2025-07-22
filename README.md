# Timeline Visualization Tool

## Run Locally

1. Create a virtual environment and activate it:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run timeline_app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click "New app", select your repo, branch (`main`), and set the main file to `timeline_app.py`.
4. Click "Deploy".

Your app will be live with a public URL! 