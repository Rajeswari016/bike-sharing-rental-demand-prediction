import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import os


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
# DARK BLUE THEME + FONT COLORS
# =========================================================
st.markdown("""
<style>

    /* =========================================
       MAIN BACKGROUND
       ========================================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #06152F 0%,
            #0A2348 50%,
            #0F3568 100%
        );
    }


    /* =========================================
       MAIN TEXT
       ========================================= */

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 900;
        color: #FFFFFF !important;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 18px;
        color: #D9E7FF !important;
        margin-bottom: 35px;
    }


    /* =========================================
       NORMAL TEXT
       ========================================= */

    p {
        color: #E5EEF9 !important;
    }

    span {
        color: #E5EEF9;
    }

    label {
        color: #FFFFFF !important;
    }


    /* =========================================
       HEADINGS
       ========================================= */

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }


    /* =========================================
       SIDEBAR
       ========================================= */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #020B1B 0%,
            #082451 50%,
            #0B356A 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }


    /* =========================================
       SIDEBAR INPUT BOXES
       ========================================= */

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #102F5C;
        color: white !important;
        border: 1px solid #4F83CC;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] input {
        color: white !important;
    }


    /* =========================================
       FEATURE CARDS
       ========================================= */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            #102C55,
            #153B70
        );

        border: 1px solid #356AA8;
        border-radius: 20px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.30);
    }


    /* =========================================
       FEATURE CARD HEADINGS
       ========================================= */

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #FFFFFF !important;
    }


    /* =========================================
       FEATURE CARD TEXT
       ========================================= */

    [data-testid="stVerticalBlockBorderWrapper"] p {
        color: #DCE8F7 !important;
    }


    /* =========================================
       PREDICTION RESULT
       ========================================= */

    [data-testid="stAlert"] {
        border-radius: 18px;
        font-size: 17px;
    }


    /* =========================================
       METRIC CARDS
       ========================================= */

    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #123A78,
            #1D4ED8
        );

        border: 1px solid #5EA0FF;
        border-radius: 18px;

        padding: 18px;

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


    /* =========================================
       BUTTON
       ========================================= */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #2563EB,
            #7C3AED
        );

        color: #FFFFFF !important;

        border: none;
        border-radius: 14px;

        padding: 13px;

        font-size: 17px;
        font-weight: 800;

        width: 100%;

        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.30);
    }


    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #3B82F6,
            #8B5CF6
        );

        color: white !important;
    }


    /* =========================================
       DATAFRAME
       ========================================= */

    [data-testid="stDataFrame"] {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 5px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.25);
    }


    /* =========================================
       INFO / SUCCESS / WARNING
       ========================================= */

    [data-testid="stAlert"] p {
        color: inherit !important;
    }


    /* =========================================
       DIVIDER
       ========================================= */

    hr {
        border-color: #35557E !important;
    }


    /* =========================================
       FOOTER
       ========================================= */

    .footer-text {
        text-align: center;
        color: #D5E3F5 !important;
        font-size: 15px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL FILES
# =========================================================

MODEL_FILE = "optimized_xgboost_model.json"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "hour_bucket_encoder.pkl"


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE
]

missing_files = []

for file_name in required_files:

    if not os.path.exists(file_name):
        missing_files.append(file_name)


if missing_files:

    st.error("❌ Required model files are missing.")

    for file_name in missing_files:
        st.warning(f"Missing file: {file_name}")

    st.info(
        "Keep app.py, optimized_xgboost_model.json, "
        "scaler.pkl and hour_bucket_encoder.pkl "
        "inside the same folder."
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

try:

    loaded_model = xgb.XGBRegressor()

    loaded_model.load_model(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    hour_bucket_encoder = joblib.load(
        ENCODER_FILE
    )

except Exception as e:

    st.error(
        "❌ Error loading model or preprocessing files."
    )

    st.code(str(e))

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

    else:
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
# MAIN TITLE
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

st.subheader("🔍 About This Prediction System")


col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.subheader("🤖 Machine Learning")

        st.write(
            "Powered by an optimized XGBoost regression "
            "model trained to predict bike rental demand."
        )


with col2:

    with st.container(border=True):

        st.subheader("🌦️ Weather Factors")

        st.write(
            "Temperature, humidity, windspeed and weather "
            "conditions are considered."
        )


with col3:

    with st.container(border=True):

        st.subheader("⏰ Time Factors")

        st.write(
            "Hour, month, weekday, peak hours and working "
            "days influence the prediction."
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚴 Input Features")

st.sidebar.markdown("---")


# =========================================================
# TIME INFORMATION
# =========================================================

st.sidebar.subheader("📅 Time Information")


yr = st.sidebar.selectbox(
    "Year",
    [2011, 2012],
    index=0
)


mnth = st.sidebar.slider(
    "Month",
    1,
    12,
    7
)


hr = st.sidebar.slider(
    "Hour",
    0,
    23,
    12
)


weekday = st.sidebar.slider(
    "Day of Week",
    0,
    6,
    3,
    help="0 = Sunday, 6 = Saturday"
)


# =========================================================
# DAY INFORMATION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📆 Day Information")


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

st.sidebar.subheader("🌦️ Weather Information")


temp_raw = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.50,
    0.01
)


atemp_raw = st.sidebar.slider(
    "Feeling Temperature",
    0.0,
    1.0,
    0.50,
    0.01
)


hum_raw = st.sidebar.slider(
    "Humidity",
    0.0,
    1.0,
    0.50,
    0.01
)


windspeed_raw = st.sidebar.slider(
    "Windspeed",
    0.0,
    1.0,
    0.20,
    0.01
)


# =========================================================
# CONDITIONS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🌤️ Conditions")


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
# PREDICT BUTTON
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

        # -------------------------------------------------
        # HOLIDAY
        # -------------------------------------------------

        holiday = (
            1
            if holiday_str == "Yes"
            else 0
        )


        # -------------------------------------------------
        # WORKING DAY
        # -------------------------------------------------

        workingday = (
            1
            if workingday_str == "Working Day"
            else 0
        )


        # -------------------------------------------------
        # WEATHER SCALING
        # -------------------------------------------------

        weather_array = np.array([
            [
                temp_raw,
                atemp_raw,
                hum_raw,
                windspeed_raw
            ]
        ])


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


        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        peak_hour = get_peak_hour(
            hr
        )


        hour_bucket = get_hour_bucket_str(
            hr
        )


        hour_bucket_encoded = encode_hour_bucket(
            hour_bucket
        )


        weekend = get_weekend(
            weekday
        )


        temp_difference = abs(
            temp_raw - atemp_raw
        )


        # -------------------------------------------------
        # SEASON ENCODING
        # -------------------------------------------------

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


        # -------------------------------------------------
        # WEATHER ENCODING
        # -------------------------------------------------

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


        # -------------------------------------------------
        # INPUT DATAFRAME
        # -------------------------------------------------

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
        # PREDICT
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
        # RESULT
        # =================================================

        st.markdown("---")

        st.header("📊 Prediction Result")


        st.success(
            f"🚴 Predicted Total Bike Rentals: "
            f"{predicted_count:,}"
        )


        # Large metric
        st.metric(
            label="🚴 Bikes Expected for Selected Hour",
            value=f"{predicted_count:,}"
        )


        # =================================================
        # SELECTED CONDITIONS
        # =================================================

        st.markdown("---")

        st.header("📌 Selected Conditions")


        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)


        with col_a:

            st.info(
                f"📅 **Year**\n\n"
                f"### {yr}"
            )


        with col_b:

            st.info(
                f"📆 **Month**\n\n"
                f"### {mnth}"
            )


        with col_c:

            st.info(
                f"⏰ **Hour**\n\n"
                f"### {hr:02d}:00"
            )


        with col_d:

            st.info(
                f"🌡️ **Temperature**\n\n"
                f"### {temp_raw:.2f}"
            )


        # =================================================
        # OTHER CONDITIONS
        # =================================================

        st.subheader("🌦️ Other Selected Conditions")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.warning(
                f"🎉 Holiday: **{holiday_str}**"
            )


        with c2:

            st.warning(
                f"💼 Working Day: **{workingday_str}**"
            )


        with c3:

            st.warning(
                f"🍂 Season: **{season_str.title()}**"
            )


        st.info(
            f"🌤️ Weather Situation: **{weathersit_str}**"
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
            "❌ Something went wrong while making the prediction."
        )

        st.code(str(e))


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