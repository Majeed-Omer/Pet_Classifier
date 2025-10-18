# 🐈 Pet Classifier - AI-Powered Pet Recognition
An intelligent web application that uses deep learning to automatically identify pet types from uploaded images. Built with Streamlit and TensorFlow, this project demonstrates practical application of transfer learning and computer vision.

---

## 🌟 Features

🎯 Instant Classification: Upload pet images and get real-time predictions
🐶 Multi-Pet Support: Recognizes dogs, cats, birds, rabbits, hamsters, and guinea pigs
📊 Confidence Scoring: Visual progress bars showing prediction accuracy
🧠 Transfer Learning: Powered by MobileNetV2 pretrained on ImageNet
💡 Smart Mapping: Intelligently maps ImageNet classes to common pet categories
🎨 Beautiful UI: Clean, modern interface with emoji indicators and color-coded results
⚡ Fast Performance: Lightweight CNN model with ~3.5M parameters for quick inference

---

## 📸 Screenshots

- https://github.com/Majeed-Omer/Pet_Classifier/blob/main/ScrrenShots/Capture-1.PNG
- https://github.com/Majeed-Omer/Pet_Classifier/blob/main/ScrrenShots/Capture-2.PNG
- https://github.com/Majeed-Omer/Pet_Classifier/blob/main/ScrrenShots/Capture-3.PNG
- https://github.com/Majeed-Omer/Pet_Classifier/blob/main/ScrrenShots/Capture-4.PNG
- https://github.com/Majeed-Omer/Pet_Classifier/blob/main/ScrrenShots/Capture-5.PNG

Main Interface
Upload any pet image and get instant classification results with confidence scores.

## Prediction Results
Large emoji display for identified pet type
Confidence percentage with visual progress bar
Detailed breakdown of all prediction scores
Color-coded confidence indicators

---

## 📖 Usage

Launch the Application: Run the Streamlit app using the command above
Upload an Image: Click on "Choose a pet image..." and select an image file (JPG, JPEG, or PNG)
View Results: The AI will analyze the image and display:

Primary pet type prediction with emoji
Confidence percentage
Visual confidence bar
Detailed breakdown of all predictions

Try Different Images: Upload multiple images to test various pet types and breeds

### Tips for Best Results

Use clear, well-lit images
Ensure the pet is the main subject of the photo
Avoid blurry or very small images
Front-facing or side profile shots work best
Images with single pets perform better than group photos

---

## 🏗️ Architecture
Model Details

Base Model: MobileNetV2
Training Data: ImageNet (1000 classes)
Parameters: ~3.5 million
Input Size: 224x224 pixels
Architecture Type: Convolutional Neural Network (CNN)
Technique: Transfer Learning

### How It Works
Image Upload: User uploads a pet image through Streamlit interface

Preprocessing: Image is resized to 224x224 and normalized for MobileNetV2

Prediction: Model predicts top-10 ImageNet classes with confidence scores

Mapping: ImageNet classes are intelligently mapped to pet categories:

Dogs: 26+ breed classes

Cats: 5 breed classes

Birds: 20+ species classes

Rabbits: 2 classes

Small pets: Hamsters and guinea pigs

Results: Aggregated scores are displayed with the highest confidence pet type

---

## 🎯 Supported Pet Types
The classifier can identify the following pet categories:

🐶 Dogs (26+ breeds including Labrador, Golden Retriever, Bulldog, Pug, Husky, etc.)
🐱 Cats (Including Tabby, Persian, Siamese, Egyptian Cat)
🐦 Birds (20+ species including Robin, Eagle, Owl, Peacock, Parrot, etc.)
🐰 Rabbits (Hare, Wood Rabbit)
🐹 Small Pets (Hamsters, Guinea Pigs)



## 🚀 How to Run Locally
Clone the repository
git clone https://github.com/Majeed-Omer/kurdish-animal-care

---

## 👨‍💻 Author

Majeed Omer Majeed

📬 Contacts

GitHub: @Majeed-Omer

LinkedIn: Majeed Omer

Email: majeedomer32@gmail.com

⭐ Show Your Support If you like this project, please give it a ⭐ on GitHub and share it with your friends!
