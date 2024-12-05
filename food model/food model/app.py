from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_filename = 'cals_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load label encoders
with open('le_food_category.pkl', 'rb') as le_file:
    le_food_category = pickle.load(le_file)

with open('le_food_item.pkl', 'rb') as le_file:
    le_food_item = pickle.load(le_file)

# Load the list of categories and items (assuming you have these lists)
food_categories = le_food_category.classes_.tolist()
food_items = le_food_item.classes_.tolist()

@app.route('/')
def home():
    return render_template('index.html', food_categories=food_categories, food_items=food_items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        food_category = request.form['FoodCategory']
        food_item = request.form['FoodItem']
        per100grams = '100'  # Assuming you have a default value here

        # Encode categorical variables
        food_category_encoded = le_food_category.transform([food_category])[0]
        food_item_encoded = le_food_item.transform([food_item])[0]

        # Process input data (ensure all features are included in the correct order)
        input_data = pd.DataFrame({
            'FoodCategory': [food_category_encoded],
            'FoodItem': [food_item_encoded],
            'per100grams': [per100grams]  # Ensure all features are included
        })

        # Make prediction using your model
        prediction = model.predict(input_data)

        # Format prediction result
        prediction_result = {
            'prediction': float(prediction[0])
        }

        return render_template('result.html', prediction=prediction_result['prediction'])

    except Exception as e:
        return render_template('result.html', prediction=str(e))

if __name__ == '__main__':
    app.run(debug=True)
