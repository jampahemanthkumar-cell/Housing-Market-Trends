
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Housing Market Trends",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Housing Market Trends & Price Intelligence")
st.caption(
    "Housing-value analysis, machine learning, "
    "geographic patterns, and price estimation"
)

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


# ---------------- DATA LOADING ----------------

@st.cache_data(show_spinner="Loading California housing dataset...")
def load_data():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()

    # The original target is measured in units of $100,000.
    df["MedianHouseValue"] = dataset.target
    df["PriceUSD"] = df["MedianHouseValue"] * 100_000

    return df


# ---------------- MODEL TRAINING ----------------

@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(data):
    X = data[FEATURES].copy()
    y = data["MedianHouseValue"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            random_state=42,
        ),
    }

    fitted_models = {}
    metric_rows = []

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        metric_rows.append({
            "Model": name,
            "MAE (USD)": mae * 100_000,
            "RMSE (USD)": rmse * 100_000,
            "R2": r2,
        })

        fitted_models[name] = pipeline

    metrics = pd.DataFrame(metric_rows).sort_values(
        "RMSE (USD)"
    ).reset_index(drop=True)

    best_name = metrics.iloc[0]["Model"]

    # Save the actual evaluation results.
    metrics.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False,
    )

    return (
        fitted_models,
        metrics,
        best_name,
        X_test,
        y_test,
    )


# ---------------- INITIALIZE APPLICATION ----------------

try:
    df = load_data()

    (
        models,
        metrics,
        best_name,
        X_test,
        y_test,
    ) = train_models(df)

    best_model = models[best_name]

except Exception as exc:
    st.error(
        "Could not load the dataset or train the models. "
        "Check the internet connection and installed libraries."
    )
    st.exception(exc)
    st.stop()


# ---------------- NAVIGATION ----------------

st.sidebar.title("🏠 Housing Intelligence")
st.sidebar.success(f"Best test RMSE: {best_name}")

page = st.sidebar.radio(
    "Navigate",
    [
        "Market Overview",
        "Price Prediction",
        "Batch Prediction",
        "Model Evaluation",
        "Explore Data",
    ],
)


# ---------------- MARKET OVERVIEW ----------------

if page == "Market Overview":
    st.subheader("California Housing Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Housing records", f"{len(df):,}")
    c2.metric(
        "Median area value",
        f"${df['PriceUSD'].median():,.0f}",
    )
    c3.metric(
        "Mean area value",
        f"${df['PriceUSD'].mean():,.0f}",
    )
    c4.metric(
        "Best model R²",
        f"{metrics.iloc[0]['R2']:.3f}",
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            px.histogram(
                df,
                x="PriceUSD",
                nbins=40,
                title="Distribution of Area-Level Median Values",
                labels={
                    "PriceUSD": "Median house value (USD)"
                },
            ),
            use_container_width=True,
        )

    with right:
        sample = df.sample(
            min(3000, len(df)),
            random_state=42,
        )

        st.plotly_chart(
            px.scatter(
                sample,
                x="MedInc",
                y="PriceUSD",
                color="HouseAge",
                opacity=0.55,
                title="House Value vs Household Income",
                labels={
                    "MedInc": "Median income ($10,000 units)",
                    "PriceUSD": "Median house value (USD)",
                    "HouseAge": "Median house age (years)",
                },
            ),
            use_container_width=True,
        )

    st.subheader("Geographic Housing-Value Patterns")

    map_df = df.sample(
        min(4000, len(df)),
        random_state=42,
    )

    st.plotly_chart(
        px.scatter(
            map_df,
            x="Longitude",
            y="Latitude",
            color="PriceUSD",
            color_continuous_scale="Viridis",
            hover_data=[
                "MedInc",
                "HouseAge",
                "PriceUSD",
            ],
            title="Housing Values Across California",
            labels={
                "PriceUSD": "Median house value (USD)"
            },
        ),
        use_container_width=True,
    )

    st.info(
        "This dataset represents historical census-area housing "
        "values. It does not provide a month-by-month price series "
        "or current property listings."
    )


# ---------------- SINGLE PRICE PREDICTION ----------------

elif page == "Price Prediction":
    st.subheader("Estimate an Area's Median House Value")

    st.caption(
        "This model estimates a census area's median house value, "
        "not the selling price of an individual property."
    )

    with st.form("housing_prediction_form"):
        left, right = st.columns(2)

        with left:
            income = st.slider(
                "Median household income ($10,000 units)",
                0.5, 15.0, 4.0, 0.1,
            )
            house_age = st.slider(
                "Median house age (years)",
                1, 52, 25,
            )
            rooms = st.slider(
                "Average rooms per household",
                1.0, 15.0, 5.0,
            )
            bedrooms = st.slider(
                "Average bedrooms per household",
                0.5, 5.0, 1.0,
            )

        with right:
            population = st.slider(
                "Area population",
                3, 35000, 1000,
            )
            occupancy = st.slider(
                "Average household occupancy",
                1.0, 10.0, 3.0,
            )
            latitude = st.slider(
                "Latitude",
                32.5, 42.0, 36.0,
            )
            longitude = st.slider(
                "Longitude",
                -124.5, -114.0, -119.0,
            )

        submitted = st.form_submit_button(
            "Estimate House Value"
        )

    if submitted:
        record = pd.DataFrame([{
            "MedInc": income,
            "HouseAge": house_age,
            "AveRooms": rooms,
            "AveBedrms": bedrooms,
            "Population": population,
            "AveOccup": occupancy,
            "Latitude": latitude,
            "Longitude": longitude,
        }])[FEATURES]

        estimated_value = max(
            0.0,
            float(best_model.predict(record)[0]) * 100_000,
        )

        st.metric(
            "Estimated area median house value",
            f"${estimated_value:,.0f}",
        )

        st.caption(
            "Educational estimate only. The dataset's target has "
            "an upper-value ceiling, so high-value areas may be "
            "underestimated."
        )


# ---------------- BATCH PREDICTION ----------------

elif page == "Batch Prediction":
    st.subheader("Batch Housing-Value Estimation")

    st.write(
        "Upload a CSV containing all eight required input "
        "features. A target column is not required."
    )

    st.download_button(
        "Download Example Input CSV",
        df[FEATURES].head(5).to_csv(
            index=False
        ).encode("utf-8"),
        file_name="housing_input_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload housing data",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)

            missing = sorted(
                set(FEATURES) - set(batch.columns)
            )

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

            elif batch.empty:
                st.error("The uploaded CSV contains no rows.")

            else:
                batch = batch[FEATURES].copy()

                for col in FEATURES:
                    batch[col] = pd.to_numeric(
                        batch[col],
                        errors="coerce",
                    )

                batch = batch.dropna(
                    subset=FEATURES,
                    how="all",
                )

                if batch.empty:
                    st.error(
                        "No usable numeric input rows were found."
                    )
                else:
                    batch["EstimatedMedianValueUSD"] = (
                        best_model.predict(
                            batch[FEATURES]
                        ) * 100_000
                    ).clip(lower=0)

                    st.dataframe(
                        batch,
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download Prediction Results",
                        batch.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="housing_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(
                f"Could not process the uploaded file: {exc}"
            )


# ---------------- MODEL EVALUATION ----------------

elif page == "Model Evaluation":
    st.subheader("Compare Regression Models")

    st.caption(
        "Metrics are calculated on a held-out test set."
    )

    st.dataframe(
        metrics.style.format({
            "MAE (USD)": "${:,.0f}",
            "RMSE (USD)": "${:,.0f}",
            "R2": "{:.3f}",
        }),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            px.bar(
                metrics,
                x="Model",
                y="RMSE (USD)",
                color="Model",
                title="RMSE — Lower Is Better",
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            px.bar(
                metrics,
                x="Model",
                y="R2",
                color="Model",
                title="R² — Higher Is Generally Better",
            ),
            use_container_width=True,
        )

    st.download_button(
        "Download Model Evaluation CSV",
        metrics.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv",
    )

    st.info(
        "MAE is the average absolute prediction error. RMSE "
        "penalizes larger errors more strongly. R² measures how "
        "much target variance is explained on the test set."
    )


# ---------------- EXPLORE DATA ----------------

elif page == "Explore Data":
    st.subheader("Explore Housing Features")

    selected_feature = st.selectbox(
        "Choose a feature",
        FEATURES,
    )

    st.plotly_chart(
        px.histogram(
            df,
            x=selected_feature,
            nbins=40,
            title=f"Distribution of {selected_feature}",
        ),
        use_container_width=True,
    )

    correlation = df[
        FEATURES + ["MedianHouseValue"]
    ].corr()

    st.subheader("Feature Correlation Analysis")

    st.plotly_chart(
        px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Housing Feature Correlation Matrix",
        ),
        use_container_width=True,
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.download_button(
        "Download Analysis Dataset",
        df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="housing_analysis_data.csv",
        mime="text/csv",
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Housing Market Trends | Educational machine-learning "
    "project | Historical census-area data, not live market data "
    "or professional property valuation advice."
)
