import joblib
from flask import Flask, request, jsonify

# Load the pre-trained model
model = joblib.load('random_forest_model_important.pkl')  # Ensure this file path is correct

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request
    data = request.get_json()  # Assumes data is sent in JSON format

    # Get the features from the input data
    features = data['features']

    # Use the model to make a prediction
    prediction = model.predict([features])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
