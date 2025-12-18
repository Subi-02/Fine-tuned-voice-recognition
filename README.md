# Fine-tuned-voice-recognition


📌 Project Overview

This project implements an audio-only speech emotion recognition system that identifies human emotions from spoken audio signals using a fine-tuned pretrained deep learning model.

The system processes raw speech input and predicts emotional states such as neutral, happy, sad, angry, fearful, disgust, and surprise, based solely on audio characteristics.

🎯 Objective

To design and train a deep learning model capable of detecting emotions from speech audio by adapting a pretrained speech representation model through fine-tuning.

🧠 Model Used
Wav2Vec2.0

Transformer-based speech representation model

Pretrained on large-scale unlabeled speech data

Fine-tuned for emotion classification using labeled emotional speech data

🛠 Technologies & Libraries

Python

PyTorch

Hugging Face Transformers

Librosa

Torchaudio

NumPy

Scikit-learn

Google Colab (GPU)

📂 Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Professional actor recordings

High-quality speech audio

8 emotion classes:

Neutral

Calm

Happy

Sad

Angry

Fearful

Disgust

Surprised

Dataset is downloaded and processed automatically within the project code.

🔄 Audio Processing Pipeline

Load .wav audio files

Resample audio to 16 kHz

Normalize audio amplitude

Extract speech representations using Wav2Vec2

Fine-tune classification layers

Predict emotion probabilities

⚙️ Fine-Tuning Approach

Pretrained Wav2Vec2 encoder as feature extractor

Classification head trained on emotional speech labels

Cross-entropy loss function

AdamW optimizer

Low learning rate to prevent overfitting

🏗 Project Structure
Audio-Emotion-Detection/
│
├── data/
│   └── dataset_download.py
│
├── preprocessing/
│   └── audio_preprocessing.py
│
├── training/
│   └── train_model.py
│
├── inference/
│   └── predict_emotion.py
│
├── requirements.txt
└── README.md

🚀 How to Run
Install Dependencies
pip install transformers datasets librosa torchaudio soundfile

Train the Model
python train_model.py

Predict Emotion
python predict_emotion.py --audio sample.wav

📊 Sample Output
Predicted Emotion: Neutral

Emotion Probabilities:
Neutral   : 0.62
Happy     : 0.18
Sad       : 0.10
Angry     : 0.06
Fearful   : 0.04

🧪 Evaluation

Accuracy measurement

Emotion-wise probability analysis

Confusion matrix visualization

💡 Applications

Speech-based sentiment analysis

Call center monitoring

Voice-based interaction systems

Behavioral and emotional analysis

🔮 Future Enhancements

Real-time microphone input

Multilingual emotion recognition

Noise-robust emotion detection

Deployment using REST APIs

Extension to video-based emotion recognition
