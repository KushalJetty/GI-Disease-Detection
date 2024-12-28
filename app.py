import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil

app = Flask(__name__)

# Load your trained model
model = load_model('model_path')
# Dataset path
dataset_path='dataset_path'
# Load class names 
class_names = sorted(os.listdir(dataset_path)) 
# Background image
bg_img = "image_url"

@app.route('/')
def index():
    return render_template('home.html',bg_img=bg_img)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        # Handle user registration
        return render_template('index.html', msg='Successfully Registered')
    return render_template('index.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
             'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             "Loss Graph"
             'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Clear the directory
        dirPath = "static/images"
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        for fileName in os.listdir(dirPath):
            os.remove(os.path.join(dirPath, fileName))

        # Retrieve and save the uploaded file
        uploaded_file = request.files['filename']
        if uploaded_file.filename == '':
            return render_template('userlog.html', msg="No file selected")
        
        file_path = os.path.join(dirPath, uploaded_file.filename)
        uploaded_file.save(file_path)
        
        # RGB Matrix
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_matrix = rgb_image.tolist()  # Convert to list for rendering in HTML

        # Normalize the RGB matrix
        normalization_matrix = (rgb_image / 255.0).tolist()  # Normalize and convert to list

        # Preprocess for model input
        def preprocess_input_image(path):
            img = load_img(path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Prediction
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            return predicted_class, confidence

         # Generate 10x10 matrix
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, (10, 10))  # Resize to 10x10
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        normalized_matrix = rgb_image / 255.0  # Normalize
        rgb_matrix = rgb_image.tolist()
        normalized_matrix = np.round(normalized_matrix, 2).tolist()  # Round values for better display

        predicted_class, confidence = predict_single_image(file_path)

        # Map predicted class to labels and treatment
        Tre = ""
        Tre1 = []
        if predicted_class == "dyed-lifted-polyps":
            Tre = "Medical Treatment"
            Tre1 = [
                "Polyp Removal: Undergo endoscopic resection if the polyp poses a risk.",
                "Post-Procedure Care: Follow the doctor's advice regarding diet and medications.",
                "Regular Screening: Schedule follow-up endoscopies to monitor for recurrence or new polyps."
            ]
        elif predicted_class == "normal-z-line":
            Tre = "No Treatment Needed"
            Tre1 = [
                "Healthy Lifestyle: Maintain a balanced diet to support overall digestive health.",
                "Routine Check-ups: Continue regular screenings as per your healthcare provider's advice.",
                "Stay Hydrated: Drink adequate water to maintain a healthy gastrointestinal tract."
            ]
        elif predicted_class == "polyps":
            Tre = "Medical Treatment"
            Tre1 = [
                "Endoscopic Removal: Remove polyps via colonoscopy to prevent potential malignancy.",
                "Lifestyle Adjustments: Avoid smoking, maintain a high-fiber diet, and reduce alcohol consumption.",
                "Periodic Monitoring: Schedule regular check-ups to detect new or recurring polyps early."
            ]
        elif predicted_class == "dyed-resection-margins":
            Tre = "Medical Treatment"
            Tre1 = [
                "Post-Surgical Care: Follow prescribed antibiotics and pain relievers as needed.",
                "Healing Monitoring: Schedule follow-ups to ensure proper healing of resection sites.",
                "Lifestyle Guidance: Maintain a diet low in irritants to support gastrointestinal recovery."
            ]
        elif predicted_class == "ulcerative-colitis":
            Tre = "Medical Treatment"
            Tre1 = [
                "Medication: Use anti-inflammatory drugs or immunosuppressants as prescribed.",
                "Dietary Adjustments: Follow a low-residue or anti-inflammatory diet to manage symptoms.",
                "Regular Monitoring: Attend follow-ups for symptom management and to prevent complications."
            ]
        elif predicted_class == "esophagitis":
            Tre = "Medical Treatment"
            Tre1 = [
                "Medication: Use antacids, proton pump inhibitors, or other prescribed medications.",
                "Dietary Changes: Avoid acidic, spicy, or hot foods to reduce irritation.",
                "Lifestyle Adjustments: Stop smoking, avoid alcohol, and elevate the head while sleeping."
            ]
        elif predicted_class == "normal-cecum":
            Tre = "No Treatment Needed"
            Tre1 = [
                "Routine Monitoring: Maintain regular health screenings as recommended.",
                "Healthy Eating: Focus on a balanced diet rich in fiber for optimal colon health.",
                "Stay Active: Engage in regular physical activity to promote gastrointestinal well-being."
            ]
        elif predicted_class == "normal-pylorus":
            Tre = "No Treatment Needed"
            Tre1 = [
                "Maintain Nutrition: Continue with a balanced diet and proper hydration.",
                "Routine Check-ups: Schedule regular health examinations as a preventive measure.",
                "Avoid Stomach Irritants: Limit spicy foods, caffeine, and alcohol for stomach health."
            ]

        # Prepare results
        accuracy = f"The predicted image is {predicted_class} with a confidence of {confidence:.2%}"

        input_image = preprocess_input_image(file_path)
        logits = model.predict(input_image)

        # Apply softmax to get probabilities
        softmax_values = tf.nn.softmax(logits[0]).numpy()

        # Get the predicted class
        predicted_class = class_names[np.argmax(softmax_values)]

        # Prepare class probabilities for display
        class_probabilities = [
            (class_name, float(prob))
            for class_name, prob in zip(class_names, softmax_values)
        ]
        
        # Generate a bar graph for class probabilities
        graph_path = os.path.join(dirPath, 'class_probabilities.png')
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, softmax_values, color='skyblue')
        plt.title('Class Probabilities')
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(graph_path)  # Save the graph in static/images
        plt.close()
        # Render results
         # Render results
        return render_template(
            'results.html',
            status=predicted_class,
            accuracy=accuracy,
            Treatment=Tre,
            Treatment1=Tre1,
            ImageDisplay=f"http://127.0.0.1:5000/static/images/{uploaded_file.filename}",
            GraphDisplay=url_for('static', filename='images/class_probabilities.png'),
            predicted_class=predicted_class,
            class_probabilities=class_probabilities,
            rgb_matrix=rgb_matrix,
            normalization_matrix=normalization_matrix
        )
    return render_template('userlog.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
