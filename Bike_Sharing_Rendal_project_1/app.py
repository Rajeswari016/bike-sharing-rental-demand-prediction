import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# FILE PATHS
# =========================================================
MODEL_FILE = BASE_DIR / "optimized_xgboost_model.json"
SCALER_FILE = BASE_DIR / "scaler.pkl"
ENCODER_FILE = BASE_DIR / "hour_bucket_encoder.pkl"


# =========================================================
# FIND DATASET
# =========================================================
possible_dataset_files = [
    "Bike_Sharing_Rental_Dataset.csv",
    "Bike_Sharing_Rental_Dataset.csvfile.csv",
    "Dataset.csv",
    "hour.csv",
    "train.csv"
]

DATA_FILE = None

for file_name in possible_dataset_files:
    file_path = BASE_DIR / file_name

    if file_path.exists():
        DATA_FILE = file_path
        break


# =========================================================
# DARK BLUE THEME
# =========================================================
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #061A33 0%,
            #082B52 50%,
            #0B3A6E 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 900;
        color: #FFFFFF !important;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 18px;
        color: #D7E7FA !important;
        margin-bottom: 30px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    p {
        color: #E6EEF8 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #020B18 0%,
            #06234A 50%,
            #0A3A6A 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #123A67 !important;
        color: #FFFFFF !important;
        border: 1px solid #4A8FD1 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            #102E53,
            #123B6D
        ) !important;

        border: 1px solid #3A72A8 !important;
        border-radius: 20px !important;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.30);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #FFFFFF !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] p {
        color: #DCE8F5 !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #123E78,
            #1D4ED8
        ) !important;

        border: 1px solid #60A5FA !important;
        border-radius: 18px !important;

        padding: 18px !important;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.30);
    }

    [data-testid="stMetricLabel"] {
        color: #DCEBFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }

    .stButton > button {
        width: 100%;

        background: linear-gradient(
            135deg,
            #2563EB,
            #7C3AED
        ) !important;

        color: #FFFFFF !important;

        border: none !important;
        border-radius: 14px !important;

        padding: 13px 20px !important;

        font-size: 17px !important;
        font-weight: 800 !important;

        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.30);
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #3B82F6,
            #8B5CF6
        ) !important;

        color: #FFFFFF !important;
    }

    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border-radius: 15px !important;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stAlert"] {
        border-radius: 16px !important;
    }

    hr {
        border-color: #3B5F86 !important;
    }

    .footer-text {
        text-align: center;
        color: #D8E5F3 !important;
        font-size: 15px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CHECK MODEL FILES
# =========================================================
required_model_files = [
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE
]

missing_model_files = [
    file_path.name
    for file_path in required_model_files
    if not file_path.exists()
]

if missing_model_files:

    st.error("❌ Required model files are missing.")

    for file_name in missing_model_files:
        st.warning(f"Missing file: {file_name}")

    st.info(
        "The model files must be in the same folder as app.py."
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================
try:

    loaded_model = xgb.XGBRegressor()

    loaded_model.load_model(
        str(MODEL_FILE)
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    hour_bucket_encoder = joblib.load(
        ENCODER_FILE
    )

except Exception as e:

    st.error(
        "❌ Error loading the model or preprocessing files."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_peak_hour(hour):

    if hour in [7, 8, 9, 17, 18, 19]:
        return 1

    return 0


def get_hour_bucket_str(hour):

    if hour in [7, 8, 9]:
        return "morning_rush"

    elif hour in [17, 18, 19]:
        return "evening_rush"

    elif 10 <= hour <= 16:
        return "daytime"

    return "night"


def get_weekend(weekday):

    if weekday in [0, 6]:
        return 1

    return 0


def encode_hour_bucket(hour_bucket):

    try:

        encoded = hour_bucket_encoder.transform(
            [hour_bucket]
        )

        return float(
            np.asarray(encoded).ravel()[0]
        )

    except Exception:

        return 0.0


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">'
    '🚴 Bike Rental Demand Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Predict hourly bike rental demand using Machine Learning '
    'and environmental conditions.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ABOUT PROJECT
# =========================================================
st.header(
    "🔍 About This Prediction System"
)

col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.subheader(
            "🤖 Machine Learning"
        )

        st.write(
            "Powered by an optimized XGBoost regression "
            "model trained to predict bike rental demand."
        )


with col2:

    with st.container(border=True):

        st.subheader(
            "🌦️ Weather Factors"
        )

        st.write(
            "Temperature, humidity, windspeed and weather "
            "conditions are considered."
        )


with col3:

    with st.container(border=True):

        st.subheader(
            "⏰ Time Factors"
        )

        st.write(
            "Hour, month, weekday, peak hours and working "
            "days influence the prediction."
        )


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title(
    "🚴 Input Features"
)

st.sidebar.markdown("---")


# =========================================================
# TIME INFORMATION
# =========================================================
st.sidebar.subheader(
    "📅 Time Information"
)

yr = st.sidebar.selectbox(
    "Year",
    [2011, 2012],
    index=0
)

mnth = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=7
)

hr = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

weekday = st.sidebar.slider(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=3,
    help="0 = Sunday, 6 = Saturday"
)


# =========================================================
# DAY INFORMATION
# =========================================================
st.sidebar.markdown("---")

st.sidebar.subheader(
    "📆 Day Information"
)

holiday_str = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)

workingday_str = st.sidebar.selectbox(
    "Working Day",
    ["No work", "Working Day"]
)


# =========================================================
# WEATHER INFORMATION
# =========================================================
st.sidebar.markdown("---")

st.sidebar.subheader(
    "🌦️ Weather Information"
)

temp_raw = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

atemp_raw = st.sidebar.slider(
    "Feeling Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

hum_raw = st.sidebar.slider(
    "Humidity",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

windspeed_raw = st.sidebar.slider(
    "Windspeed",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)


# =========================================================
# CONDITIONS
# =========================================================
st.sidebar.markdown("---")

st.sidebar.subheader(
    "🌤️ Conditions"
)

season_str = st.sidebar.selectbox(
    "Season",
    [
        "spring",
        "summer",
        "fall",
        "winter"
    ]
)

weathersit_str = st.sidebar.selectbox(
    "Weather Situation",
    [
        "Clear",
        "Mist",
        "Light Snow",
        "Heavy Rain"
    ]
)


# =========================================================
# PREDICTION BUTTON
# =========================================================
st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "🚀 Predict Rental Demand",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

    try:

        holiday = (
            1
            if holiday_str == "Yes"
            else 0
        )

        workingday = (
            1
            if workingday_str == "Working Day"
            else 0
        )


        # =================================================
        # WEATHER SCALING
        # =================================================
        weather_array = np.array(
            [[
                temp_raw,
                atemp_raw,
                hum_raw,
                windspeed_raw
            ]]
        )

        scaled_features = scaler.transform(
            weather_array
        )

        temp = float(
            scaled_features[0][0]
        )

        atemp = float(
            scaled_features[0][1]
        )

        hum = float(
            scaled_features[0][2]
        )

        windspeed = float(
            scaled_features[0][3]
        )


        # =================================================
        # FEATURE ENGINEERING
        # =================================================
        peak_hour = get_peak_hour(hr)

        hour_bucket = get_hour_bucket_str(hr)

        hour_bucket_encoded = encode_hour_bucket(
            hour_bucket
        )

        weekend = get_weekend(weekday)

        temp_difference = abs(
            temp_raw - atemp_raw
        )


        # =================================================
        # SEASON ENCODING
        # =================================================
        season_spring = (
            1
            if season_str == "spring"
            else 0
        )

        season_summer = (
            1
            if season_str == "summer"
            else 0
        )

        season_winter = (
            1
            if season_str == "winter"
            else 0
        )


        # =================================================
        # WEATHER ENCODING
        # =================================================
        weathersit_heavy_rain = (
            1
            if weathersit_str == "Heavy Rain"
            else 0
        )

        weathersit_light_snow = (
            1
            if weathersit_str == "Light Snow"
            else 0
        )

        weathersit_mist = (
            1
            if weathersit_str == "Mist"
            else 0
        )


        # =================================================
        # INPUT DATAFRAME
        # =================================================
        input_data = pd.DataFrame(
            [[
                yr,
                mnth,
                hr,
                holiday,
                weekday,
                workingday,
                temp,
                atemp,
                hum,
                windspeed,
                peak_hour,
                hour_bucket_encoded,
                weekend,
                temp_difference,
                season_spring,
                season_summer,
                season_winter,
                weathersit_heavy_rain,
                weathersit_light_snow,
                weathersit_mist
            ]],
            columns=[
                "yr",
                "mnth",
                "hr",
                "holiday",
                "weekday",
                "workingday",
                "temp",
                "atemp",
                "hum",
                "windspeed",
                "Peak_Hour",
                "hour_bucket",
                "Weekend",
                "Temp_Difference",
                "season_spring",
                "season_summer",
                "season_winter",
                "weathersit_Heavy Rain",
                "weathersit_Light Snow",
                "weathersit_Mist"
            ]
        )


        # =================================================
        # MODEL PREDICTION
        # =================================================
        prediction = loaded_model.predict(
            input_data
        )

        predicted_count = max(
            0,
            int(
                round(
                    float(prediction[0])
                )
            )
        )


        # =================================================
        # PREDICTION RESULT
        # =================================================
        st.markdown("---")

        st.header(
            "📊 Prediction Result"
        )

        st.success(
            f"🚴 Predicted Total Bike Rentals: "
            f"{predicted_count:,}"
        )

        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "🚴 Predicted Rentals",
                f"{predicted_count:,}"
            )


        with r2:

            st.metric(
                "⏰ Selected Hour",
                f"{hr:02d}:00"
            )


        with r3:

            st.metric(
                "🌤️ Weather",
                weathersit_str
            )


        # =================================================
        # SELECTED CONDITIONS
        # =================================================
        st.markdown("---")

        st.header(
            "📌 Selected Conditions"
        )

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "📅 Year",
                yr
            )


        with c2:

            st.metric(
                "📆 Month",
                mnth
            )


        with c3:

            st.metric(
                "⏰ Hour",
                f"{hr:02d}:00"
            )


        with c4:

            st.metric(
                "🌡️ Temperature",
                f"{temp_raw:.2f}"
            )


        # =================================================
        # OTHER CONDITIONS
        # =================================================
        st.subheader(
            "🌦️ Other Selected Conditions"
        )

        o1, o2, o3 = st.columns(3)


        with o1:

            st.info(
                f"🎉 Holiday: **{holiday_str}**"
            )


        with o2:

            st.info(
                f"💼 Working Day: **{workingday_str}**"
            )


        with o3:

            st.info(
                f"🍂 Season: **{season_str.title()}**"
            )


        st.info(
            f"🌤️ Weather Situation: "
            f"**{weathersit_str}**"
        )


        # =================================================
        # INPUT DATA
        # =================================================
        st.markdown("---")

        st.header(
            "📋 Input Data Used for Prediction"
        )

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # PREDICTION INFORMATION
        # =================================================
        st.markdown("---")

        st.subheader(
            "💡 Prediction Information"
        )

        st.info(
            "The prediction is generated using the trained "
            "XGBoost machine learning model based on the "
            "selected time, weather and environmental "
            "conditions."
        )


    except Exception as e:

        st.error(
            "❌ Something went wrong while making "
            "the prediction."
        )

        st.code(
            str(e)
        )


# =========================================================
# DATA VISUALIZATIONS
# =========================================================
st.markdown("---")

st.header(
    "📊 Bike Rental Data Visualizations"
)


# =========================================================
# DATASET CHECK
# =========================================================
if DATA_FILE is None:

    st.warning(
        "⚠️ Dataset file not found."
    )

    st.info(
        "Please keep the dataset in the same folder as app.py."
    )

else:

    try:

        # =================================================
        # LOAD DATASET
        # =================================================
        viz_df = pd.read_csv(
            DATA_FILE
        )


        # =================================================
        # CLEAN DATA
        # =================================================
        viz_df = viz_df.replace(
            "?",
            np.nan
        )


        numeric_columns = [
            "hr",
            "mnth",
            "yr",
            "holiday",
            "weekday",
            "workingday",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "cnt",
            "season",
            "weathersit"
        ]


        for column in numeric_columns:

            if column in viz_df.columns:

                viz_df[column] = pd.to_numeric(
                    viz_df[column],
                    errors="coerce"
                )


        # =================================================
        # DATASET LOADED
        # =================================================
        st.success(
            f"✅ Dataset loaded successfully: "
            f"{DATA_FILE.name}"
        )


        # =================================================
        # DATASET OVERVIEW
        # =================================================
        st.subheader(
            "📈 Dataset Overview"
        )


        v1, v2, v3, v4 = st.columns(4)


        with v1:

            st.metric(
                "📚 Total Records",
                f"{len(viz_df):,}"
            )


        with v2:

            st.metric(
                "📊 Total Columns",
                len(viz_df.columns)
            )


        with v3:

            st.metric(
                "🔢 Numeric Columns",
                viz_df.select_dtypes(
                    include=np.number
                ).shape[1]
            )


        with v4:

            if "cnt" in viz_df.columns:

                total_rentals = (
                    viz_df["cnt"]
                    .dropna()
                    .sum()
                )

                st.metric(
                    "🚴 Total Rentals",
                    f"{int(total_rentals):,}"
                )


        # =================================================
        # HOURLY DEMAND
        # =================================================
        if {
            "hr",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "⏰ Average Bike Rental Demand by Hour"
            )


            hourly_df = viz_df[
                ["hr", "cnt"]
            ].dropna()


            hourly_demand = (
                hourly_df
                .groupby(
                    "hr",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            hourly_demand.columns = [
                "Hour",
                "Average Rentals"
            ]


            st.line_chart(
                hourly_demand.set_index("Hour"),
                use_container_width=True
            )


            st.caption(
                "Average rental demand for each hour of the day."
            )


        # =================================================
        # MONTHLY DEMAND
        # =================================================
        if {
            "mnth",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "📅 Average Bike Rental Demand by Month"
            )


            monthly_df = viz_df[
                ["mnth", "cnt"]
            ].dropna()


            monthly_demand = (
                monthly_df
                .groupby(
                    "mnth",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            monthly_demand.columns = [
                "Month",
                "Average Rentals"
            ]


            st.bar_chart(
                monthly_demand.set_index("Month"),
                use_container_width=True
            )


        # =================================================
        # SEASON-WISE DEMAND
        # =================================================
        if {
            "season",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "🍂 Season-wise Bike Rental Demand"
            )


            season_df = viz_df[
                ["season", "cnt"]
            ].dropna()


            season_demand = (
                season_df
                .groupby(
                    "season",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            season_names = {
                1: "Spring",
                2: "Summer",
                3: "Fall",
                4: "Winter"
            }


            season_demand["Season"] = (
                season_demand["season"]
                .map(season_names)
            )


            season_demand = season_demand[
                ["Season", "cnt"]
            ]


            season_demand.columns = [
                "Season",
                "Average Rentals"
            ]


            st.bar_chart(
                season_demand.set_index("Season"),
                use_container_width=True
            )


        # =================================================
        # WORKING DAY VS NON-WORKING DAY
        # =================================================
        if {
            "workingday",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "💼 Working Day vs Non-Working Day"
            )


            working_df = viz_df[
                ["workingday", "cnt"]
            ].dropna()


            working_demand = (
                working_df
                .groupby(
                    "workingday",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            working_demand["Day Type"] = (
                working_demand["workingday"]
                .map({
                    0: "Non-Working Day",
                    1: "Working Day"
                })
            )


            working_demand = working_demand[
                ["Day Type", "cnt"]
            ]


            working_demand.columns = [
                "Day Type",
                "Average Rentals"
            ]


            st.bar_chart(
                working_demand.set_index("Day Type"),
                use_container_width=True
            )


            st.caption(
                "Average bike rental demand on working "
                "days compared with non-working days."
            )


        # =================================================
        # WEATHER SITUATION
        # =================================================
        if {
            "weathersit",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "🌦️ Weather Situation vs Rental Demand"
            )


            weather_df = viz_df[
                ["weathersit", "cnt"]
            ].dropna()


            weather_demand = (
                weather_df
                .groupby(
                    "weathersit",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            weather_names = {
                1: "Clear",
                2: "Mist",
                3: "Light Snow",
                4: "Heavy Rain"
            }


            weather_demand["Weather"] = (
                weather_demand["weathersit"]
                .map(weather_names)
            )


            weather_demand = weather_demand[
                ["Weather", "cnt"]
            ]


            weather_demand.columns = [
                "Weather",
                "Average Rentals"
            ]


            st.bar_chart(
                weather_demand.set_index("Weather"),
                use_container_width=True
            )


        # =================================================
        # TEMPERATURE
        # =================================================
        if {
            "temp",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "🌡️ Temperature vs Rental Demand"
            )


            temp_df = viz_df[
                ["temp", "cnt"]
            ].dropna()


            if len(temp_df) > 0:

                temp_df["Temperature Range"] = pd.cut(
                    temp_df["temp"],
                    bins=10
                )


                temp_demand = (
                    temp_df
                    .groupby(
                        "Temperature Range",
                        observed=True,
                        as_index=False
                    )["cnt"]
                    .mean()
                    .round(0)
                )


                temp_demand["Temperature Range"] = (
                    temp_demand[
                        "Temperature Range"
                    ].astype(str)
                )


                temp_demand.columns = [
                    "Temperature Range",
                    "Average Rentals"
                ]


                st.line_chart(
                    temp_demand.set_index(
                        "Temperature Range"
                    ),
                    use_container_width=True
                )


        # =================================================
        # HUMIDITY
        # =================================================
        if {
            "hum",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "💧 Humidity vs Rental Demand"
            )


            humidity_df = viz_df[
                ["hum", "cnt"]
            ].dropna()


            if len(humidity_df) > 0:

                humidity_df["Humidity Range"] = pd.cut(
                    humidity_df["hum"],
                    bins=10
                )


                humidity_demand = (
                    humidity_df
                    .groupby(
                        "Humidity Range",
                        observed=True,
                        as_index=False
                    )["cnt"]
                    .mean()
                    .round(0)
                )


                humidity_demand["Humidity Range"] = (
                    humidity_demand[
                        "Humidity Range"
                    ].astype(str)
                )


                humidity_demand.columns = [
                    "Humidity Range",
                    "Average Rentals"
                ]


                st.line_chart(
                    humidity_demand.set_index(
                        "Humidity Range"
                    ),
                    use_container_width=True
                )


        # =================================================
        # WINDSPEED
        # =================================================
        if {
            "windspeed",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "💨 Windspeed vs Rental Demand"
            )


            wind_df = viz_df[
                ["windspeed", "cnt"]
            ].dropna()


            if len(wind_df) > 0:

                wind_df["Windspeed Range"] = pd.cut(
                    wind_df["windspeed"],
                    bins=10
                )


                wind_demand = (
                    wind_df
                    .groupby(
                        "Windspeed Range",
                        observed=True,
                        as_index=False
                    )["cnt"]
                    .mean()
                    .round(0)
                )


                wind_demand["Windspeed Range"] = (
                    wind_demand[
                        "Windspeed Range"
                    ].astype(str)
                )


                wind_demand.columns = [
                    "Windspeed Range",
                    "Average Rentals"
                ]


                st.line_chart(
                    wind_demand.set_index(
                        "Windspeed Range"
                    ),
                    use_container_width=True
                )


        # =================================================
        # PEAK HOURS
        # =================================================
        if {
            "hr",
            "cnt"
        }.issubset(viz_df.columns):

            st.subheader(
                "🔥 Peak Hours vs Normal Hours"
            )


            peak_df = viz_df[
                ["hr", "cnt"]
            ].dropna().copy()


            peak_df["Peak_Hour"] = (
                peak_df["hr"]
                .apply(get_peak_hour)
            )


            peak_demand = (
                peak_df
                .groupby(
                    "Peak_Hour",
                    as_index=False
                )["cnt"]
                .mean()
                .round(0)
            )


            peak_demand["Hour Type"] = (
                peak_demand["Peak_Hour"]
                .map({
                    0: "Normal Hours",
                    1: "Peak Hours"
                })
            )


            peak_demand = peak_demand[
                ["Hour Type", "cnt"]
            ]


            peak_demand.columns = [
                "Hour Type",
                "Average Rentals"
            ]


            st.bar_chart(
                peak_demand.set_index("Hour Type"),
                use_container_width=True
            )


        # =================================================
        # RENTAL SUMMARY
        # =================================================
        if "cnt" in viz_df.columns:

            st.subheader(
                "📊 Rental Demand Summary"
            )


            clean_cnt = (
                pd.to_numeric(
                    viz_df["cnt"],
                    errors="coerce"
                )
                .dropna()
            )


            if len(clean_cnt) > 0:

                total_rentals = int(
                    clean_cnt.sum()
                )

                average_rentals = round(
                    clean_cnt.mean(),
                    2
                )

                maximum_rentals = int(
                    clean_cnt.max()
                )

                minimum_rentals = int(
                    clean_cnt.min()
                )


                s1, s2, s3, s4 = st.columns(4)


                with s1:

                    st.metric(
                        "🚴 Total Rentals",
                        f"{total_rentals:,}"
                    )


                with s2:

                    st.metric(
                        "📊 Average Rentals",
                        f"{average_rentals:,}"
                    )


                with s3:

                    st.metric(
                        "⬆️ Maximum Rentals",
                        f"{maximum_rentals:,}"
                    )


                with s4:

                    st.metric(
                        "⬇️ Minimum Rentals",
                        f"{minimum_rentals:,}"
                    )


        # =================================================
        # DATASET PREVIEW
        # =================================================
        st.subheader(
            "👀 Dataset Preview"
        )


        st.dataframe(
            viz_df.head(10),
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ Visualization section error."
        )

        st.code(
            str(e)
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div class="footer-text">

    🚴 <b>Bike Rental Demand Prediction</b>

    <br>

    Built with Python • XGBoost • Streamlit • Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
