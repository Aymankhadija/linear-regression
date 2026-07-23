import pickle
import gradio as gr
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_price(data_mb, call_minutes, sms_count, validity_days, internet_speed_mbps):
    features = np.array([[data_mb, call_minutes, sms_count, validity_days, internet_speed_mbps]])
    prediction = model.predict(features)[0]
    return f"Estimated Package Price: {prediction:.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Data (MB)", value=7000),
        gr.Number(label="Call Minutes", value=700),
        gr.Number(label="SMS Count", value=3000),
        gr.Number(label="Validity (Days)", value=30),
        gr.Number(label="Internet Speed (Mbps)", value=70),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Mobile Package Price Predictor",
    description="Enter package details to predict the mobile package price using a Linear Regression model.",
)

if __name__ == "__main__":
    demo.launch()