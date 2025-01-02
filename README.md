# GI Disease Detection

AI-powered web application for detecting gastrointestinal diseases from medical images.

## Features

- Image upload for gastrointestinal scan analysis
- Deep learning model for disease detection 
- Confidence scores and predictions
- Treatment recommendations
- User authentication and management

## Requirements

### System
- Windows 10, macOS, or Ubuntu
- Minimum 4 GB RAM, 1 GHz processor

### Software
- Python 3.x
- Required libraries:
```bash
pip install numpy opencv-python tensorflow flask matplotlib
```
- SQLite for user data management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KushalJetty/GI-Disease-Detection.git
cd GI-Disease-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install python-dotenv

```

3. Set up SQLite (optional)

4. Configure environment variables if needed (create `.env` file)

## Usage

1. Start the application:
```bash
python app.py
```

2. Access via `http://localhost:5000/`

3. Upload images and view predictions

## Project Structure
```
gi-disease-detection/
├── static/           # Static files (images, CSS)
├── templates/        # HTML templates
├── app.py           # Main application
├── requirements.txt # Dependencies
├── README.md        # Documentation
└── model.h5         # Trained model
```

## Dataset

Uses Kvasir Dataset v2 containing gastrointestinal tract images. Available on Kaggle.

## Contributing

1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature-name
```

3. Commit changes:
```bash
git commit -m "Add your message"
```

4. Push and open pull request

## License

MIT License
