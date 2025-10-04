# 🚀 NASA Space Apps Challenge 2025 - Exoplanet Detection AI

## 📋 Project Overview
Advanced Machine Learning system for automated exoplanet detection using NASA's space mission data (Kepler, K2, TESS).

## 🎯 Team Information
- **Team Name**: Galactic Vanguard
- **Team Owner**: Mohsen Keshavarzian
- **Team Members**: 
  - Mahla Jafarpour
  - Janyar Rakhshanfar

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/aiolearn04/Galactic-Vanguard.git
cd "Galactic-Vanguard-main"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## 📊 Features
- **Multi-Mission Support**: Kepler, K2, and TESS datasets
- **Automatic Dataset Detection**: Smart identification of mission type
- **Advanced ML Models**: LightGBM with feature engineering
- **Habitability Analysis**: Custom scoring system for planet habitability
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Reports**: Detailed analysis and visualizations

## 🗂️ Project Structure
```
Hunting for Exoplanets with AI/
├── main.py                  # Main Streamlit application
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup configuration
├── run.bat                 # Windows run script
├── run.sh                  # Linux/Mac run script
├── README.md               # Project documentation
├── data/                   # Dataset files
│   ├── K2.csv
│   ├── kepler.csv
│   └── TOI.csv
├── models/                 # Trained ML models
│   ├── kepler.pkl
│   ├── k2.pkl
│   ├── toi.pkl
│   ├── k2_input_requirements.json
│   └── toi_feature_mapping.json
├── img/                    # Images and visualizations
│   ├── banner.PNG
│   ├── class_balancing_comparison.png
│   ├── complete_model_comparison.png
│   ├── ensemble_model_performance.png
│   ├── final_model_comparison.png
│   ├── new_features_distribution.png
│   ├── smart_model_comparison.png
│   └── toi_class_distribution.png
└── NASA_Space_Apps_2025_Project_Report.docx
```

## 🚀 Usage
1. **Upload CSV File**: Upload exoplanet data from any NASA mission
2. **Automatic Detection**: System automatically identifies mission type
3. **Model Prediction**: Get exoplanet classification results
4. **Habitability Analysis**: Analyze planet habitability potential
5. **View Reports**: Access comprehensive analysis and visualizations

## 📈 Model Performance
- **Kepler Model**: 97.5% accuracy
- **K2 Model**: 98.6% accuracy  
- **TESS Model**: 71.4% accuracy

## 🔬 Technical Details
- **Algorithms**: LightGBM with feature engineering
- **Feature Engineering**: Custom features for each mission
- **Data Preprocessing**: Smart missing value handling
- **Class Balancing**: Advanced techniques for imbalanced data

## 📄 Documentation
- **Project Report**: `NASA_Space_Apps_2025_Project_Report.docx`
- **Technical Details**: See individual model training scripts

## 🤝 Contributing
This project was developed for the NASA Space Apps Challenge 2025.

## 📜 License
This project is part of the NASA Space Apps Challenge 2025.

## 📞 Contact
- **Team Owner**: Mohsen Keshavarzian
- **Project**: A World Away: Hunting for Exoplanets with AI
