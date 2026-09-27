import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(page_title="AeroInsight - Airline Satisfaction Analytics", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('../data/train.csv')
    df = df.drop(columns=['Unnamed: 0', 'id'])
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
    return df

@st.cache_resource
def load_model():
    with open('../models_saved.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()
saved = load_model()
model = saved['model']
encoders = saved['label_encoders']
features = saved['features']

st.title("✈️ AeroInsight — Airline Passenger Satisfaction Analytics")
st.caption("Independent project using a public airline passenger satisfaction dataset.")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Segment Analysis", "🔮 Predict Satisfaction"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    sat_pct = (df['satisfaction'] == 'satisfied').mean() * 100
    avg_delay = df['Departure Delay in Minutes'].mean()

    col1.metric("Total Passengers", f"{total:,}")
    col2.metric("Overall Satisfaction", f"{sat_pct:.1f}%")
    col3.metric("Avg Departure Delay", f"{avg_delay:.1f} min")
    col4.metric("Classes", df['Class'].nunique())

    st.subheader("Satisfaction by Class")
    class_sat = df.groupby('Class')['satisfaction'].value_counts(normalize=True).unstack() * 100
    fig = px.bar(class_sat, barmode='group', title="Satisfaction % by Travel Class")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Explore by Segment")
    segment = st.selectbox("Choose a segment to analyze", ['Customer Type', 'Type of Travel', 'Class'])
    seg_data = df.groupby(segment)['satisfaction'].value_counts(normalize=True).unstack() * 100
    fig2 = px.bar(seg_data, barmode='group', title=f"Satisfaction by {segment}")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Delay Impact on Satisfaction")
    df['delay_bucket'] = pd.cut(
        df['Departure Delay in Minutes'],
        bins=[-1, 0, 15, 60, 2000],
        labels=['No Delay', '1-15 min', '16-60 min', '60+ min']
    )
    delay_sat = df.groupby('delay_bucket', observed=True)['satisfaction'].apply(
        lambda x: (x == 'satisfied').mean() * 100
    )
    fig3 = px.bar(delay_sat, title="Satisfaction % by Departure Delay")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Predict Passenger Satisfaction")
    st.write("Enter passenger details to predict satisfaction likelihood.")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        cust_type = st.selectbox("Customer Type", ['Loyal Customer', 'disloyal Customer'])
        age = st.slider("Age", 7, 85, 35)
    with c2:
        travel_type = st.selectbox("Type of Travel", ['Business travel', 'Personal Travel'])
        travel_class = st.selectbox("Class", ['Business', 'Eco', 'Eco Plus'])
        distance = st.slider("Flight Distance", 50, 5000, 1000)
    with c3:
        wifi = st.slider("Inflight Wifi Service (0-5)", 0, 5, 3)
        entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 3)
        dep_delay = st.slider("Departure Delay (min)", 0, 300, 0)

    if st.button("Predict Satisfaction"):
        input_data = {
            'Gender': encoders['Gender'].transform([gender])[0],
            'Customer Type': encoders['Customer Type'].transform([cust_type])[0],
            'Age': age,
            'Type of Travel': encoders['Type of Travel'].transform([travel_type])[0],
            'Class': encoders['Class'].transform([travel_class])[0],
            'Flight Distance': distance,
            'Inflight wifi service': wifi,
            'Ease of Online booking': 3,
            'Food and drink': 3,
            'Seat comfort': 3,
            'Inflight entertainment': entertainment,
            'On-board service': 3,
            'Leg room service': 3,
            'Baggage handling': 3,
            'Checkin service': 3,
            'Inflight service': 3,
            'Cleanliness': 3,
            'Departure Delay in Minutes': dep_delay,
            'Arrival Delay in Minutes': dep_delay
        }
        input_df = pd.DataFrame([input_data])[features]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        if pred == 1:
            st.success(f"✅ Predicted: SATISFIED (confidence: {prob[1]*100:.1f}%)")
        else:
            st.error(f"❌ Predicted: NOT SATISFIED (confidence: {prob[0]*100:.1f}%)")