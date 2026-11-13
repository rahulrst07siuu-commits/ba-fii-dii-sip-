# SIP Shield Predictor

A simple Streamlit app for the Domestic SIP Shield Effect project.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `SIP_Shield_Additional_3_Variables_Data.xlsx` — historical dataset

## Run locally (Windows)

1. Install Python 3.10 or newer.
2. Extract this ZIP file.
3. Open Command Prompt in the extracted `SIP-Shield-Predictor` folder.
4. Install packages:

   ```bash
   py -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   py -m streamlit run app.py
   ```

6. Open the local URL shown in the terminal, usually http://localhost:8501.
7. Upload the included Excel workbook when prompted.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and the Excel workbook to the repository root.
3. On Streamlit Community Cloud, create an app from that repository.
4. Set the main file path to `app.py` and deploy.

## Model

The app uses standard scikit-learn Multiple Linear Regression:

Nifty return = beta0 + beta1*SIP + beta2*FII_Selling
               + beta3*Previous_Return + beta4*Previous_3M_Volatility + error

SIP is scaled by 1,000 (₹ crore). The app cleans missing/infinite rows,
uses a chronological 80/20 holdout for evaluation, and refits the final
model on all clean historical observations.

Predictions are historical-model estimates, not guaranteed market forecasts
or investment advice.
