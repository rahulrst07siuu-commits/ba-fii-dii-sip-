import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="SIP Shield Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stButton > button {
        background-color: #C93649;
        color: white;
        border-radius: 8px;
        min-height: 44px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #A92739;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 SIP Shield Predictor")
st.caption(
    "Domestic SIP Shield Effect | Multiple Linear Regression | "
    "Nifty 50 monthly return estimation"
)

st.write(
    "Upload the historical Excel dataset. The app cleans the data, "
    "trains a regression model, displays its test performance and "
    "estimates a return from the inputs you provide."
)

# -------------------------------------------------
# Load and clean historical data
# -------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    raw = pd.read_excel(
        uploaded_file,
        sheet_name="Original Data + Variables"
    )

    required_columns = [
        "Month",
        "Nifty 50 Return (%)",
        "SIP Inflow (cr)",
        "FII Selling (0/1)",
        "Previous Month Nifty Return (%)",
        "Previous 3-Month Return Volatility (%)"
    ]

    missing = [c for c in required_columns if c not in raw.columns]
    if missing:
        raise ValueError(
            "The Excel sheet is missing these columns: " + ", ".join(missing)
        )

    data = raw[required_columns].copy()
    data["Month"] = pd.to_datetime(data["Month"], errors="coerce")

    numeric_columns = required_columns[1:]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    rows_before = len(data)
    data = data.dropna().copy()

    # Basic validity checks for the variables used by this model
    data = data[
        (data["SIP Inflow (cr)"] >= 0) &
        (data["FII Selling (0/1)"].isin([0, 1])) &
        (data["Previous 3-Month Return Volatility (%)"] >= 0)
    ]

    data = data.sort_values("Month").reset_index(drop=True)
    removed_rows = rows_before - len(data)

    if len(data) < 10:
        raise ValueError(
            "Fewer than 10 valid rows remain after cleaning. "
            "Please check the uploaded workbook."
        )

    return data, removed_rows


# -------------------------------------------------
# Model training and chronological evaluation
# -------------------------------------------------
def train_model(data):
    model_data = data.copy()
    model_data["SIP_1000"] = model_data["SIP Inflow (cr)"] / 1000

    feature_columns = [
        "SIP_1000",
        "FII Selling (0/1)",
        "Previous Month Nifty Return (%)",
        "Previous 3-Month Return Volatility (%)"
    ]

    X = model_data[feature_columns]
    y = model_data["Nifty 50 Return (%)"]

    # Keep time order: first 80% for training, final 20% for testing
    split_index = int(len(model_data) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_test) < 2:
        raise ValueError("Not enough test observations for evaluation.")

    test_model = LinearRegression()
    test_model.fit(X_train, y_train)
    test_predictions = test_model.predict(X_test)

    performance = {
        "R²": r2_score(y_test, test_predictions),
        "MAE": mean_absolute_error(y_test, test_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_predictions)),
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }

    # Refit final model using all clean historical observations
    final_model = LinearRegression()
    final_model.fit(X, y)

    return final_model, performance, feature_columns


# -------------------------------------------------
# Main interface
# -------------------------------------------------
st.subheader("Historical dataset")

uploaded_file = st.file_uploader(
    "Upload SIP Shield Excel workbook (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info(
        "Upload the included Excel workbook to begin. "
        "Use the sheet named 'Original Data + Variables'."
    )
    st.stop()

try:
    data, removed_rows = load_data(uploaded_file)
    model, performance, feature_columns = train_model(data)
except Exception as error:
    st.error(f"Could not prepare the model: {error}")
    st.stop()

st.success("Dataset cleaned and model trained successfully.")

m1, m2, m3 = st.columns(3)
m1.metric("Valid observations", f"{len(data)}")
m2.metric("Rows removed", f"{removed_rows}")
m3.metric(
    "Historical period",
    f"{data['Month'].min():%b %Y} – {data['Month'].max():%b %Y}"
)

with st.expander("View cleaned data"):
    st.dataframe(data, use_container_width=True)

st.divider()

# -------------------------------------------------
# Single prediction
# -------------------------------------------------
st.subheader("Single Nifty 50 return estimate")
st.write("Enter the monthly values that will be used by the model.")

col1, col2, col3 = st.columns(3)

with col1:
    sip = st.number_input(
        "Monthly SIP inflow (₹ crore)",
        min_value=0.0,
        value=15000.0,
        step=500.0
    )

with col2:
    fii_status = st.selectbox(
        "FII market status",
        ["FII Selling", "FII Buying / Non-selling"]
    )

with col3:
    previous_return = st.number_input(
        "Previous month Nifty return (%)",
        value=1.0,
        step=0.1,
        format="%.2f"
    )

previous_volatility = st.number_input(
    "Previous 3-month return volatility (%)",
    min_value=0.0,
    value=3.0,
    step=0.1,
    format="%.2f"
)

if st.button("Predict Nifty Return", use_container_width=False):
    fii_selling = 1 if fii_status == "FII Selling" else 0

    input_data = pd.DataFrame([{
        "SIP_1000": sip / 1000,
        "FII Selling (0/1)": fii_selling,
        "Previous Month Nifty Return (%)": previous_return,
        "Previous 3-Month Return Volatility (%)": previous_volatility
    }], columns=feature_columns)

    estimated_return = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction result")
    st.metric(
        "Estimated Nifty 50 monthly return",
        f"{estimated_return:.2f}%"
    )
    st.caption(
        "This is a model-based estimate from historical relationships. "
        "It is not a guaranteed market forecast or investment recommendation."
    )

# -------------------------------------------------
# Batch prediction
# -------------------------------------------------
st.divider()
st.subheader("Batch prediction (optional)")
st.write(
    "Upload a CSV or Excel file with these columns: "
    "SIP Inflow (cr), FII Selling (0/1), "
    "Previous Month Nifty Return (%), and "
    "Previous 3-Month Return Volatility (%)."
)

batch_file = st.file_uploader(
    "Upload batch input file",
    type=["csv", "xlsx"],
    key="batch_upload"
)

if batch_file is not None:
    try:
        if batch_file.name.lower().endswith(".csv"):
            batch = pd.read_csv(batch_file)
        else:
            batch = pd.read_excel(batch_file)

        required_batch = [
            "SIP Inflow (cr)",
            "FII Selling (0/1)",
            "Previous Month Nifty Return (%)",
            "Previous 3-Month Return Volatility (%)"
        ]

        missing_batch = [c for c in required_batch if c not in batch.columns]
        if missing_batch:
            st.error(
                "Batch file is missing columns: " + ", ".join(missing_batch)
            )
        else:
            clean_batch = batch.copy()
            for col in required_batch:
                clean_batch[col] = pd.to_numeric(
                    clean_batch[col], errors="coerce"
                )

            clean_batch = clean_batch.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            clean_batch = clean_batch[
                (clean_batch["SIP Inflow (cr)"] >= 0) &
                (clean_batch["FII Selling (0/1)"].isin([0, 1])) &
                (clean_batch["Previous 3-Month Return Volatility (%)"] >= 0)
            ].copy()

            if clean_batch.empty:
                st.warning("No valid rows remain in the batch file.")
            else:
                batch_X = pd.DataFrame({
                    "SIP_1000": clean_batch["SIP Inflow (cr)"] / 1000,
                    "FII Selling (0/1)": clean_batch["FII Selling (0/1)"],
                    "Previous Month Nifty Return (%)":
                        clean_batch["Previous Month Nifty Return (%)"],
                    "Previous 3-Month Return Volatility (%)":
                        clean_batch["Previous 3-Month Return Volatility (%)"]
                }, columns=feature_columns)

                clean_batch["Predicted Nifty Return (%)"] = (
                    model.predict(batch_X)
                )

                st.dataframe(clean_batch, use_container_width=True)

                csv_data = clean_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download batch predictions",
                    data=csv_data,
                    file_name="SIP_Shield_Predictions.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Batch prediction error: {error}")

# -------------------------------------------------
# Model performance
# -------------------------------------------------
st.divider()
st.subheader("Model performance")

p1, p2, p3 = st.columns(3)
p1.metric("Test R²", f"{performance['R²']:.3f}")
p2.metric("Test MAE", f"{performance['MAE']:.3f}")
p3.metric("Test RMSE", f"{performance['RMSE']:.3f}")

st.caption(
    f"Chronological holdout: {performance['train_rows']} training rows "
    f"and {performance['test_rows']} testing rows. "
    "The final prediction model is refitted on all clean historical data."
)

with st.expander("Regression equation"):
    st.latex(
        r"R_t = \beta_0 + \beta_1 SIP_t + "
        r"\beta_2 FIISELL_t + \beta_3 R_{t-1} + "
        r"\beta_4 VOL_{t-1,3} + \epsilon_t"
    )
    coef_table = pd.DataFrame({
        "Variable": feature_columns,
        "Coefficient": model.coef_
    })
    st.dataframe(coef_table, use_container_width=True)
    st.write("Intercept:", round(model.intercept_, 4))
