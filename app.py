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
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Load your trained model
model = load_model('model_path')
# Dataset path
dataset_path='dataset_path'
# Load class names 
class_names = sorted(os.listdir(dataset_path)) 

@app.route('/header')
def header():
    return render_template('header.html')

@app.route('/footer')
def footer():
    return render_template('footer.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        cursor.execute("SELECT name, password FROM user WHERE name = ?", (name,))
        result = cursor.fetchone()

        if result is None:
            return render_template('login.html', msg='User not found.')
        else:
            stored_hash = result[1]
            if check_password_hash(stored_hash, password):
                # Successful login - You might want to store user session information here
                return render_template('home.html')
            else:
                return render_template('login.html', msg='Incorrect password.')

        connection.close()

    return render_template('login.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        # Hash the password for security
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Connect to the database
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        # Insert user data into the database
        try:
            cursor.execute("""
                INSERT INTO user (name, email, phone, password)
                VALUES (?, ?, ?, ?)
            """, (name, email, phone, hashed_password))
            connection.commit()
            return render_template('login.html', msg='Registration successful!')
        except sqlite3.Error as e:
            connection.rollback()
            return render_template('login.html', msg=f'Error during registration: {e}')
        finally:
            connection.close()

    # Activate registration form section in login.html on GET requests
    return render_template('login.html', active_form='form2')

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
             'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             "Loss Graph",
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
        
        # Read the image and convert to RGB
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract Red, Green, and Blue channels
        red_matrix = rgb_image[:, :, 0].tolist()  # Red channel
        green_matrix = rgb_image[:, :, 1].tolist()  # Green channel
        blue_matrix = rgb_image[:, :, 2].tolist()  # Blue channel
        
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
            red_matrix=red_matrix, 
            green_matrix=green_matrix, 
            blue_matrix=blue_matrix, 
            normalization_matrix=normalization_matrix
        )
    return render_template('userlog.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)