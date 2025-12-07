⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/ved-1234/human_activity_detection.git

cd human_activity_detection

2️⃣ Create a virtual environment 
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Application
python app.py


Then open your browser and visit:

http://127.0.0.1:5000

🎥 Sample Input

You can test the model using the provided sample video:

sample_video.mp4

📊 Model Training

To train or retrain the model:

Open lstm_train.ipynb

Run cells step by step

Save the trained model weights

📈 Output

Predicts and classifies human activities from video input

Displays results via Flask-based UI
