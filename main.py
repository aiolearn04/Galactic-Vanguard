import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

#=======================
# NASA Space Apps Challenge 2025
# Exoplanet Detection AI System
# Developed by: Mohsen
#=======================


st.set_page_config(
    page_title="Exoplanet Detection AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    /* Light theme styling */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Light theme overrides */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    
    .stFileUploader > div > div {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

#=======================
# Model Management
#=======================

def get_available_models() -> Dict[str, str]:
    
    model_paths = {
        'Kepler': 'models/kepler.pkl',
        'K2': 'models/k2.pkl',
        'TESS': 'models/toi.pkl'
    }
    
    available_models = {}
    for mission_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            available_models[mission_name] = model_path
        else:
            st.error(f"❌ {mission_name} model not found at {model_path}")
    
    return available_models

def identify_dataset_mission(df: pd.DataFrame) -> str:
    
    columns = df.columns.tolist()
    
    if 'kepid' in columns and 'koi_disposition' in columns:
        return 'Kepler'
    elif 'pl_name' in columns and 'k2_name' in columns:
        return 'K2' 
    elif 'toi' in columns and 'tfopwg_disp' in columns:
        return 'TESS'
    else:
        return 'Unknown'

#=======================
# Feature Engineering
#=======================

def create_advanced_features(df: pd.DataFrame, mission_type: str) -> pd.DataFrame:
    
    enhanced_df = df.copy()
    
    if mission_type == 'K2':
        _add_k2_features(enhanced_df)
    elif mission_type == 'TESS':
        _add_tess_features(enhanced_df)
    
    return enhanced_df

def _add_k2_features(df: pd.DataFrame) -> None:
    
    if 'pl_rade' in df.columns and 'st_rad' in df.columns:
        df['planet_star_ratio'] = df['pl_rade'] / df['st_rad']
    
    if 'pl_orbper' in df.columns and 'pl_rade' in df.columns:
        df['orbital_density'] = df['pl_orbper'] / (df['pl_rade'] ** 3)
    
    if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
        df['calculated_density'] = df['pl_masse'] / (df['pl_rade'] ** 3)
    
   
    missing_columns = ['pl_orbsmax', 'pl_masse', 'pl_massj', 'pl_insol', 'pl_eqt', 'pl_orbincl', 'st_lum']
    for col in missing_columns:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
        else:
            df[f'{col}_missing'] = 0

def _add_tess_features(df: pd.DataFrame) -> None:
    
    if 'pl_rade' in df.columns and 'st_rad' in df.columns:
        df['planet_star_ratio'] = df['pl_rade'] / df['st_rad']
    
    if 'pl_orbper' in df.columns and 'pl_rade' in df.columns:
        df['orbital_density'] = df['pl_orbper'] / (df['pl_rade'] ** 3)
    
    if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
        df['calculated_density'] = df['pl_masse'] / (df['pl_rade'] ** 3)
    else:
        df['calculated_density'] = 0
    
    if 'pl_trandep' in df.columns and 'pl_trandurh' in df.columns:
        df['transit_efficiency'] = df['pl_trandep'] / df['pl_trandurh']
    
    if 'pl_insol' in df.columns and 'pl_eqt' in df.columns:
        df['habitability_index'] = df['pl_insol'] * df['pl_eqt']
    
    if 'st_rad' in df.columns and 'st_teff' in df.columns:
        df['stellar_luminosity'] = (df['st_rad'] ** 2) * (df['st_teff'] / 5778) ** 4
    
    if 'pl_trandep' in df.columns and 'pl_trandurh' in df.columns:
        df['transit_depth_duration_ratio'] = df['pl_trandep'] / df['pl_trandurh']

#=======================
# Model Prediction
#=======================

def run_exoplanet_classification(model_path: str, df: pd.DataFrame, mission_type: str) -> Optional[Dict[str, Any]]:
    
    try:
        model_data = joblib.load(model_path)
        
       
        model_config = _get_model_config(model_data, mission_type)
        if not model_config:
            return None
        
        
        enhanced_df = create_advanced_features(df, mission_type)
        features, targets = _prepare_training_data(enhanced_df, model_config, mission_type)
        
        if features is None or targets is None:
            return None
        
        
        predictions = _make_predictions(model_config['model'], features, model_config.get('scaler'))
        
        
        metrics = _calculate_performance_metrics(targets, predictions, model_config['class_names'])
        
        return {
            'predictions': predictions,
            'true_labels': targets,
            'class_names': model_config['class_names'],
            'metrics': metrics,
            'available_features': len(model_config['feature_columns']),
            'total_samples': len(predictions)
        }
        
    except Exception as e:
        st.error(f"❌ Classification failed: {str(e)}")
        return None

def _get_model_config(model_data: Dict, mission_type: str) -> Optional[Dict]:
    
    try:
        if mission_type == 'Kepler':
            return {
                'model': model_data['model'],
                'feature_columns': model_data['feature_columns'],
                'target_column': 'koi_disposition',
                'scaler': None,
                'label_encoder': None,
                'class_names': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
            }
        elif mission_type == 'K2':
            return {
                'model': model_data['model'],
                'feature_columns': model_data['training_features'],
                'target_column': 'disposition',
                'scaler': None,
                'label_encoder': None,
                'class_names': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
            }
        elif mission_type == 'TESS':
            return {
                'model': model_data['ensemble'],
                'feature_columns': model_data['feature_columns'],
                'target_column': 'tfopwg_disp',
                'scaler': model_data['scaler'],
                'label_encoder': model_data['label_encoder'],
                'class_names': model_data['label_encoder'].classes_ if 'label_encoder' in model_data else ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
            }
        return None
    except KeyError as e:
        st.error(f"❌ Missing model configuration: {str(e)}")
        return None

def _prepare_training_data(df: pd.DataFrame, config: Dict, mission_type: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    
    try:
        
        available_features = [col for col in config['feature_columns'] if col in df.columns]
        if not available_features:
            st.error("❌ No matching features found in dataset")
            return None, None
        
        features = df[available_features].fillna(df[available_features].median())
        
        
        if mission_type == 'K2':
            targets = df[config['target_column']].replace('REFUTED', 'FALSE POSITIVE')
        elif mission_type == 'TESS':
            class_mapping = {
                'PC': 'CANDIDATE', 'APC': 'CANDIDATE', 
                'CP': 'CONFIRMED', 'KP': 'CONFIRMED',
                'FP': 'FALSE_POSITIVE', 'FA': 'FALSE_POSITIVE'
            }
            targets = df[config['target_column']].map(class_mapping).dropna()
            features = features.iloc[:len(targets)]
        else:
            targets = df[config['target_column']]
        
        
        if config['label_encoder'] is not None:
           
            targets_encoded = config['label_encoder'].transform(targets)
        else:
            
            le = LabelEncoder()
            targets_encoded = le.fit_transform(targets)
        
        return features, targets_encoded
    except Exception as e:
        st.error(f"❌ Data preparation failed: {str(e)}")
        return None, None

def _make_predictions(model: Any, features: pd.DataFrame, scaler: Optional[Any]) -> np.ndarray:
    
    if scaler is not None:
        features_scaled = scaler.transform(features)
        return model.predict(features_scaled)
    else:
        predictions_proba = model.predict(features.values)
        return np.argmax(predictions_proba, axis=1)

def _calculate_performance_metrics(true_labels: np.ndarray, predictions: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

#=======================
# Habitability Analysis
#=======================

def analyze_planet_habitability(df: pd.DataFrame, mission_type: str) -> List[Dict[str, Any]]:
    
    habitability_scores = []
    
    for idx, planet in df.iterrows():
        score, contributing_factors = _calculate_planet_habitability_score(planet)
        habitability_scores.append({
            'habitability_score': score,
            'factors': contributing_factors
        })
    
    return habitability_scores

def _calculate_planet_habitability_score(planet: pd.Series) -> Tuple[float, List[str]]:
    
    total_score = 0.0
    contributing_factors = []
    
    
    if 'pl_eqt' in planet.index and pd.notna(planet['pl_eqt']):
        temp = planet['pl_eqt']
        if 200 <= temp <= 350:  # Habitable temperature range
            temp_score = 1 - abs(temp - 288) / 100  # 288K is Earth-like
            total_score += temp_score * 0.3
            contributing_factors.append(f"Temperature: {temp:.1f}K")
    
    # Insolation factor (30% weight)
    if 'pl_insol' in planet.index and pd.notna(planet['pl_insol']):
        insol = planet['pl_insol']
        if 0.5 <= insol <= 2.0:  # Habitable insolation range
            insol_score = 1 - abs(insol - 1.0) / 1.5
            total_score += insol_score * 0.3
            contributing_factors.append(f"Insolation: {insol:.2f}")
    
    # Size factor (20% weight)
    if 'pl_rade' in planet.index and pd.notna(planet['pl_rade']):
        radius = planet['pl_rade']
        if 0.8 <= radius <= 2.0:  # Earth-like to super-Earth
            size_score = 1 - abs(radius - 1.0) / 1.2
            total_score += size_score * 0.2
            contributing_factors.append(f"Radius: {radius:.2f} Earth")
    
    # Orbital period factor (20% weight)
    if 'pl_orbper' in planet.index and pd.notna(planet['pl_orbper']):
        period = planet['pl_orbper']
        if 200 <= period <= 500:  # Habitable orbital period
            period_score = 1 - abs(period - 365) / 200
            total_score += period_score * 0.2
            contributing_factors.append(f"Period: {period:.1f} days")
    
    # Normalize score to 0-100
    final_score = min(100, max(0, total_score * 100))
    return final_score, contributing_factors

#=======================
# Visualization Functions
#=======================

def create_habitability_visualizations(habitability_scores: List[Dict[str, Any]]) -> None:
    
    habitability_values = [score['habitability_score'] for score in habitability_scores]
    
    col1, col2 = st.columns(2)
    
    with col1:
        _create_habitability_histogram(habitability_values)
    
    with col2:
        _create_habitability_categories_chart(habitability_values)

def _create_habitability_histogram(habitability_values: List[float]) -> None:
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(habitability_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Habitability Score (%)')
    ax.set_ylabel('Number of Planets')
    ax.set_title('Distribution of Habitability Scores')
    ax.axvline(x=70, color='red', linestyle='--', label='High Habitability (70%+)')
    ax.legend()
    st.pyplot(fig)

def _create_habitability_categories_chart(habitability_values: List[float]) -> None:
    
    high_hab = sum(1 for score in habitability_values if score >= 70)
    medium_hab = sum(1 for score in habitability_values if 40 <= score < 70)
    low_hab = sum(1 for score in habitability_values if score < 40)
    
    categories = ['High Habitability\n(70%+)', 'Medium Habitability\n(40-70%)', 'Low Habitability\n(<40%)']
    counts = [high_hab, medium_hab, low_hab]
    colors = ['green', 'orange', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Planets')
    ax.set_title('Habitability Categories')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    st.pyplot(fig)

#=======================
# Main Application
#=======================

def main():
    
    try:
        st.image("img/banner.PNG", use_column_width=True)
    except:
        st.markdown('<h1 class="main-header">🚀 Exoplanet Detection AI</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Machine Learning for Exoplanet Discovery</p>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">🔧 System Status</h2>', unsafe_allow_html=True)
    
    
    available_models = get_available_models()
    
    if available_models:
        col1, col2, col3 = st.columns(3)
        for i, (mission_name, model_path) in enumerate(available_models.items()):
            with [col1, col2, col3][i % 3]:
                st.markdown(f'<div class="success-box"><h4>✅ {mission_name} Model</h4><p>Ready for Analysis</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box"><h4>❌ No Models Found</h4><p>Please ensure model files are in the models/ directory</p></div>', unsafe_allow_html=True)
        return
    
    
    st.markdown("""
    <div class="info-box">
    <h3>📊 Supported Datasets</h3>
    <ul>
        <li><strong>Kepler Mission:</strong> Long-term monitoring of stars</li>
        <li><strong>K2 Mission:</strong> Extended Kepler observations</li>
        <li><strong>TESS Mission:</strong> Transiting Exoplanet Survey Satellite</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">📁 Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing exoplanet data from NASA missions"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            
            mission_type = identify_dataset_mission(df)
            st.success(f"✅ Detected dataset type: {mission_type}")
            
            
            st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col4:
                st.metric("Dataset Type", mission_type)
            
            
            st.markdown("#### 📋 Sample Data")
            sample_data = []
            for i in range(min(10, len(df))):
                row_data = {}
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        row_data[col] = f"{df.iloc[i][col]:.4f}" if pd.notna(df.iloc[i][col]) else "N/A"
                    else:
                        row_data[col] = str(df.iloc[i][col]) if pd.notna(df.iloc[i][col]) else "N/A"
                sample_data.append(row_data)
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
            
            
            st.markdown("### 🤖 Model Prediction & Analysis")
            
            
            if mission_type in available_models:
                model_path = available_models[mission_type]
                st.info(f"🔍 Running {mission_type} model analysis...")
                
                
                prediction_results = run_exoplanet_classification(model_path, df, mission_type)
                
                if prediction_results:
                    
                    st.markdown("#### 📊 Model Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{prediction_results['metrics']['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{prediction_results['metrics']['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{prediction_results['metrics']['recall']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{prediction_results['metrics']['f1']:.3f}")
                    
                    
                    st.markdown("#### 📈 Per-Class Performance")
                    class_names = prediction_results['class_names']
                    precision_per_class = prediction_results['metrics']['precision_per_class']
                    recall_per_class = prediction_results['metrics']['recall_per_class']
                    f1_per_class = prediction_results['metrics']['f1_per_class']
                    
                    metrics_data = []
                    for i, class_name in enumerate(class_names):
                        metrics_data.append([
                            class_name,
                            f"{precision_per_class[i]:.3f}",
                            f"{recall_per_class[i]:.3f}",
                            f"{f1_per_class[i]:.3f}"
                        ])
                    
                    metrics_df = pd.DataFrame(metrics_data, columns=['Class', 'Precision', 'Recall', 'F1-Score'])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    
                    st.markdown("#### 🔄 Confusion Matrix")
                    cm = confusion_matrix(prediction_results['true_labels'], prediction_results['predictions'])
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    
                    
                    st.markdown("#### 📋 Prediction Summary")
                    unique, counts = np.unique(prediction_results['predictions'], return_counts=True)
                    prediction_summary = []
                    for i, (class_idx, count) in enumerate(zip(unique, counts)):
                        class_name = class_names[class_idx]
                        percentage = (count / len(prediction_results['predictions'])) * 100
                        prediction_summary.append({
                            'Class': class_name,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    summary_df = pd.DataFrame(prediction_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                else:
                    st.error("❌ Model prediction failed. Please check your data format.")
            else:
                st.warning(f"⚠️ No model available for {mission_type} mission type.")
            
            
            st.markdown("### 🌍 Habitability Analysis")
            habitability_scores = analyze_planet_habitability(df, mission_type)
            
            
            habitability_data = []
            for i, score_data in enumerate(habitability_scores):
                planet_info = {
                    'pl_name': df.iloc[i].get('pl_name', f'Planet_{i+1}') if 'pl_name' in df.columns else f'Planet_{i+1}',
                    'hostname': df.iloc[i].get('hostname', f'Star_{i+1}') if 'hostname' in df.columns else f'Star_{i+1}',
                    'habitability_score': f"{score_data['habitability_score']:.1f}%",
                    'habitability_factors': ', '.join(score_data['factors'][:3]) if score_data['factors'] else 'No data available'
                }
                habitability_data.append(planet_info)
            
            habitability_df = pd.DataFrame(habitability_data)
            st.dataframe(habitability_df, use_container_width=True)
            
            
            st.markdown("#### 📊 Habitability Score Distribution")
            create_habitability_visualizations(habitability_scores)
            
            
            st.markdown("#### 🌟 Most Habitable Planets")
            
            st.markdown("""
            **How We Calculate Habitability Score:**
            
            Our habitability assessment is based on four key factors that determine a planet's potential to support life:
            
            1. **Temperature (30% weight)**: We evaluate the equilibrium temperature (pl_eqt) to determine if it falls within the habitable range of 200-350K (Earth-like: 288K)
            
            2. **Insolation (30% weight)**: We analyze the stellar insolation (pl_insol) to check if it's within 0.5-2.0 times Earth's insolation (optimal: 1.0)
            
            3. **Planet Size (20% weight)**: We consider the planet radius (pl_rade) to ensure it's between 0.8-2.0 Earth radii (Earth-like: 1.0)
            
            4. **Orbital Period (20% weight)**: We examine the orbital period (pl_orbper) to verify it's within 200-500 days (Earth-like: 365 days)
            
            Each factor contributes to a normalized score (0-100%), with higher scores indicating greater habitability potential. The final score represents the planet's overall suitability for life as we know it.
            """)
            
            habitable_planets = []
            for i, score_data in enumerate(habitability_scores):
                if score_data['habitability_score'] >= 50:
                    planet_info = {
                        'Planet': df.iloc[i].get('pl_name', f'Planet_{i+1}') if 'pl_name' in df.columns else f'Planet_{i+1}',
                        'Host Star': df.iloc[i].get('hostname', f'Star_{i+1}') if 'hostname' in df.columns else f'Star_{i+1}',
                        'Habitability Score': f"{score_data['habitability_score']:.1f}%",
                        'Key Factors': ', '.join(score_data['factors'][:3])
                    }
                    habitable_planets.append(planet_info)
            
            if habitable_planets:
                habitable_planets.sort(key=lambda x: float(x['Habitability Score'].replace('%', '')), reverse=True)
                habitable_df = pd.DataFrame(habitable_planets[:5])
                st.dataframe(habitable_df, use_container_width=True)
            else:
                st.info("No planets found with significant habitability potential in this dataset.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Welcome to Exoplanet Detection AI</h3>
        <p>This advanced AI system can automatically identify exoplanets from NASA's space mission data and analyze their habitability potential.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>🔍 Automatic dataset type detection (Kepler, K2, TESS)</li>
            <li>🤖 AI-powered exoplanet classification</li>
            <li>🌍 Habitability analysis with scoring</li>
            <li>📊 Interactive visualizations</li>
            <li>📈 Performance metrics and statistics</li>
        </ul>
        <p>Simply upload a CSV file containing exoplanet data to get started!</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()