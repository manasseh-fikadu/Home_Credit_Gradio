import gradio as gr
import requests
import shap
import pandas as pd
import joblib
from matplotlib import pyplot as plt

# Load the model and test set for SHAP
model = joblib.load('./catboost_reduced.joblib')
test_set = joblib.load('./x_test.joblib')

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

# Function to get prediction from API and explain it


def get_prediction(_):
    # Call the API
    response = requests.get('https://home-credit.onrender.com/random/')
    data = response.json()

    # Get the prediction and probability
    prediction = data['prediction']
    probability = data['probability']

    # Get the data used for the prediction
    input_data = pd.DataFrame(data['data'])

    # Get the SHAP values for the prediction
    shap_values = explainer.shap_values(input_data)

    # Create a SHAP summary plot and save it
    shap.summary_plot(shap_values, input_data, show=False)
    plt.tight_layout()
    plt.savefig('shap_plot.png')

    if prediction == 1:
        prediction = 'Client with payment difficulties'
    else:
        prediction = 'All other cases'

    probability = str(probability) + '%'

    # Return the prediction, probability, data, and SHAP plot
    return prediction, probability, input_data, 'shap_plot.png'


# Create a Gradio interface
iface = gr.Interface(fn=get_prediction,
                     title="Credit Risk Prediction",
                     description="Predict if a client will have payment difficulties. Press the generate button to generate a prediction.",
                     inputs=[],
                     outputs=[
                         gr.outputs.Label(label="Prediction"),
                         gr.outputs.Label(label="Probability"),
                         gr.outputs.Dataframe(type='pandas', label="Data"),
                         gr.outputs.Image(type="filepath", label="SHAP Plot")
                     ],
                     interpretation="default")

# Launch the app
iface.launch(share=True)
