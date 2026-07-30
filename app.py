import streamlit as st
import pandas as pd
import pickle
import matplotlib as plt



# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
score = saved["score"]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📱 Mobile Package Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
    padding:20px;
    border-radius:15px;
    background:linear-gradient(90deg,#4facfe,#00f2fe);
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

.big{
    font-size:35px;
    font-weight:bold;
    color:#1E88E5;
}

.metric{
    text-align:center;
    padding:10px;
    border-radius:10px;
    background:#f1f3f6;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------

st.markdown('<p class="title">📱 Mobile Package Price Prediction</p>',
unsafe_allow_html=True)

st.write("### Predict the expected price of a mobile package using Machine Learning.")

# ---------------- SIDEBAR ----------------

st.sidebar.header("📥 Enter Package Details")

data_mb = st.sidebar.slider(
    "Internet Data (MB)",
    500,
    20000,
    5000,
    500
)

call_minutes = st.sidebar.slider(
    "Call Minutes",
    50,
    5000,
    500
)

sms = st.sidebar.slider(
    "SMS Count",
    0,
    10000,
    1000
)

validity = st.sidebar.slider(
    "Validity (Days)",
    1,
    365,
    30
)

speed = st.sidebar.slider(
    "Internet Speed (Mbps)",
    5,
    200,
    50
)

# ---------------- DATAFRAME ----------------

new_data = pd.DataFrame({
    "Data_MB":[data_mb],
    "Call_Minutes":[call_minutes],
    "SMS_Count":[sms],
    "Validity_Days":[validity],
    "Internet_Speed_Mbps":[speed]
})

# ---------------- BUTTON ----------------

if st.button("🚀 Predict Package Price", use_container_width=True):

    prediction = model.predict(new_data)[0]

    col1,col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        <div class="card">
        <h2>💰 Predicted Package Price</h2>
        """,unsafe_allow_html=True)

        st.markdown(
            f'<p class="big">Rs. {prediction:,.2f}</p>',
            unsafe_allow_html=True
        )

        st.success("Prediction completed successfully!")

        st.markdown("</div>",unsafe_allow_html=True)
        st.header("📊 Model Performance")
        st.metric("R² Score", f"{score:.4f}")
        st.metric("Model Score", f"{score*100:.2f}%")
    with col2:

        st.metric("📶 Data",f"{data_mb} MB")
        st.metric("📞 Minutes",call_minutes)
        st.metric("💬 SMS",sms)
        st.metric("📅 Days",validity)
        st.metric("⚡ Speed",f"{speed} Mbps")

    st.divider()

    st.subheader("📋 Input Summary")

    st.dataframe(new_data,use_container_width=True)

    st.divider()

    st.subheader("📊 Package Features")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        ["Data","Minutes","SMS","Days","Speed"],
        [data_mb,call_minutes,sms,validity,speed]
    )

    plt.xticks(rotation=0)

    st.pyplot(fig)

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-Learn")
