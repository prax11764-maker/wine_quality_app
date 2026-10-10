# Wine Quality Predictor — Web App

A Streamlit web app that turns your Wine Quality Prediction notebook into a live,
interactive tool. Enter a wine's chemical properties, get an instant predicted
quality score.

## Files in this folder
- `winequality.csv` — the dataset
- `train_and_save.py` — trains the final model (with class-imbalance handling) and saves it
- `app.py` — the Streamlit web app itself
- `requirements.txt` — the packages needed to run everything

## Run it on your own computer

1. Put all files in this folder in the same directory.
2. Open Anaconda Prompt, navigate to that folder:
   ```
   cd path\to\this\folder
   ```
3. Install what's needed:
   ```
   pip install -r requirements.txt
   ```
4. Train and save the model (only needs to be done once):
   ```
   python train_and_save.py
   ```
   This creates 4 new files: `wine_model.pkl`, `wine_scaler.pkl`, `wine_label_encoder.pkl`, `wine_feature_names.pkl`
5. Launch the app:
   ```
   streamlit run app.py
   ```
   Your browser will automatically open to `localhost:8501` with the live app.

## Put it on the internet for free (so you can share a real link)

1. Create a free GitHub account if you don't have one, and create a new repository
   (e.g. `wine-quality-app`).
2. Upload these files to that repository: `app.py`, `train_and_save.py`,
   `requirements.txt`, `winequality.csv`, and the 4 `.pkl` files you generated
   in step 4 above (so the app doesn't need to retrain itself online).
3. Go to https://streamlit.io/cloud and sign in with your GitHub account.
4. Click "New app", pick your repository, and set the main file to `app.py`.
5. Click "Deploy". After a minute or two, you'll get a public URL like
   `https://your-app-name.streamlit.app` — this works on any device, and you
   can put this exact link on your resume and LinkedIn.

## What to say about this in an interview

"I didn't just train the model in a notebook — I also built a Streamlit web
app on top of it, so anyone can input a wine's properties and get a live
prediction, and I deployed it publicly so it's actually usable, not just code
sitting in a notebook."
