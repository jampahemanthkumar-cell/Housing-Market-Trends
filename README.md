
# Housing Market Trends & Price Prediction

An interactive machine-learning project for analyzing housing
patterns and estimating area-level median house values.

## Project Objective

Explore historical California housing data, identify patterns
associated with house values, compare regression models, and
estimate median values using area characteristics.

## Key Features

- Interactive housing market overview
- Geographic housing-value visualization
- Individual value estimation
- Batch CSV predictions
- Random Forest regression
- Extra Trees regression
- Histogram Gradient Boosting regression
- Model evaluation using MAE, RMSE, and R-squared
- Feature distributions and correlation analysis
- Downloadable CSV reports

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

## Dataset

The project uses the California Housing dataset available
through scikit-learn.

Official documentation:
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

The dataset contains census-area characteristics such as
median income, house age, rooms, population, latitude, and
longitude.

The original target represents median house value in units
of $100,000.

The data represents historical area-level values, not live
property listings or a monthly historical price series.

## Project Structure

    Housing-Market-Trends/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── data/
    │   └── README.md
    ├── models/
    │   └── .gitkeep
    └── reports/
        └── README.md

## Run Locally

Install Python 3.10 or later.

Clone the repository:

    git clone https://github.com/YOUR-USERNAME/Housing-Market-Trends.git

Open the project folder:

    cd Housing-Market-Trends

Install dependencies:

    pip install -r requirements.txt

Run the dashboard:

    streamlit run app.py

The dataset is downloaded automatically by scikit-learn when
the application first loads. An internet connection is needed.

## Model Evaluation

The application compares three regression models using a
held-out test set.

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R-squared: Explained variance score

The application generates actual evaluation results when run.
No performance score is guaranteed in advance.

## Limitations

The model estimates historical census-area median house values.
It does not predict exact individual-property sale prices or
current market prices.

The dataset has an upper-value ceiling, which can limit
predictions for expensive housing areas.

## Author

Add your name here.

## Internship

CODETECH Internship Project
