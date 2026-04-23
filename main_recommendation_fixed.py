import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
try:
    from skill_gap_analyzer import analyze_skill_gaps
    from resource_matcher import get_resources
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure skill_gap_analyzer.py and resource_matcher.py exist")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load trained model with error handling
try:
    model = joblib.load('career_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

# Exact 42 training features from X_train.csv
TRAINING_FEATURES = [
    'age', 'tenth_percentage', 'twelfth_or_diploma_percentage', 'math_score', 
    'programming_score', 'communication_score', 'logical_reasoning_score', 
    'creativity_score', 'leadership_score', 'teamwork_score', 
    'problem_solving_score', 'coding_interest', 'design_interest', 
    'data_interest', 'business_interest', 'security_interest', 
    'cloud_interest', 'certifications_count', 'project_count', 
    'gender_Female', 'gender_Male', 'gender_Other', 
    'preferred_work_type_Analytical', 'preferred_work_type_Creative', 
    'preferred_work_type_Management', 'preferred_work_type_Technical', 
    'preferred_domain_AI-ML', 'preferred_domain_Cloud', 'preferred_domain_Data', 
    'preferred_domain_Security', 'preferred_domain_Web', 
    'personality_type_Analytical', 'personality_type_Creative', 
    'personality_type_Leader', 'personality_type_Logical', 
    'internship_experience_No', 'internship_experience_Yes', 
    'favorite_subject_DBMS', 'favorite_subject_Math', 
    'favorite_subject_Python', 'favorite_subject_Statistics', 
    'favorite_subject_Web'
]

# Required raw input columns (25)
REQUIRED_RAW_COLS = [
    'age', 'gender', 'tenth_percentage', 'twelfth_or_diploma_percentage',
    'math_score', 'programming_score', 'communication_score', 
    'logical_reasoning_score', 'creativity_score', 'leadership_score',
    'teamwork_score', 'problem_solving_score', 'coding_interest',
    'design_interest', 'data_interest', 'business_interest',
    'security_interest', 'cloud_interest', 'preferred_work_type',
    'preferred_domain', 'personality_type', 'certifications_count',
    'project_count', 'internship_experience', 'favorite_subject'
]

CATEGORICAL_COLS = [
    'gender', 'preferred_work_type', 'preferred_domain', 
    'personality_type', 'internship_experience', 'favorite_subject'
]

def validate_user_data(user_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate user data keys, types, and ranges."""
    missing = [col for col in REQUIRED_RAW_COLS if col not in user_data]
    if missing:
        return False, f"Missing columns: {missing}"
    
    errors = []
    for col, val in user_data.items():
        if col == 'age':
            try:
                fval = float(val)
                if not 15 <= fval <= 100:
                    errors.append(f"{col}: must be 15-100")
            except:
                errors.append(f"{col}: must be numeric 15-100")
        elif col in ['certifications_count', 'project_count', 'coding_interest',
                   'design_interest', 'data_interest', 'business_interest',
                   'security_interest', 'cloud_interest']:
            try:
                fval = float(val)
                if not 0 <= fval <= 10:
                    errors.append(f"{col}: must be 0-10")
            except:
                errors.append(f"{col}: must be numeric 0-10")
        elif col in ['tenth_percentage', 'twelfth_or_diploma_percentage']:
            try:
                fval = float(val)
                if not 0 <= fval <= 100:
                    errors.append(f"{col}: must be 0-100")
            except:
                errors.append(f"{col}: must be numeric 0-100")
        elif col in ['math_score', 'programming_score', 'communication_score',
                     'logical_reasoning_score', 'creativity_score',
                     'leadership_score', 'teamwork_score', 'problem_solving_score']:
            try:
                fval = float(val)
                if not 0 <= fval <= 10:
                    errors.append(f"{col}: must be 0-10")
            except:
                errors.append(f"{col}: must be numeric 0-10")
        elif col in CATEGORICAL_COLS:
            if not isinstance(val, str):
                errors.append(f"{col}: must be string")
    
    if errors:
        return False, "; ".join(errors)
    return True, "Valid"

# Dynamic CAREERS from model classes
try:
    label_encoder = joblib.load('label_encoder.pkl')
    CAREERS = {i: label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))}
    logger.info("CAREERS mapping ready from label_encoder.pkl")
except:
    logger.warning("Using fallback CAREERS mapping")
    CAREERS = {}

def preprocess_user_input(user_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw user data dict to features matching expected training shapes."""
    is_valid, msg = validate_user_data(user_data)
    if not is_valid:
        raise ValueError(f"Invalid input data: {msg}")
    
    user_df = pd.DataFrame([user_data])
    
    # Convert all numeric cols to float
    numeric_cols = [col for col in REQUIRED_RAW_COLS if col not in CATEGORICAL_COLS]
    for col in numeric_cols:
        user_df[col] = pd.to_numeric(user_df[col], errors='coerce')
    
    # Use model's expected features if available
    expected_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else TRAINING_FEATURES
    
    # Only dummy encode if expected model features demand dummy columns
    needs_dummies = any(col not in expected_features for col in CATEGORICAL_COLS)
    if needs_dummies:
        user_df = pd.get_dummies(user_df, columns=CATEGORICAL_COLS, dummy_na=False)

    user_encoded = user_df.reindex(columns=expected_features, fill_value=0.0)
    
    # Verify shape
    if not hasattr(model, 'feature_names_in_') and user_encoded.shape[1] != 42:
        logger.warning(f"Expected 42 features, got {user_encoded.shape[1]}")
    
    logger.info(f"Preprocessed to shape: {user_encoded.shape}")
    return user_encoded

def full_recommendation_pipeline(user_data_dict: Dict[str, Any], user_skills: List[str]) -> Dict[str, Any]:
    """Complete recommendation pipeline with full error handling."""
    try:
        # Preprocess
        user_features = preprocess_user_input(user_data_dict)
        
        # Predict
        prediction_proba = model.predict_proba(user_features)
        prediction = model.predict(user_features)[0]
        confidence = float(prediction_proba[0].max() * 100)
        
        if isinstance(prediction, (int, np.integer)):
            predicted_career = CAREERS.get(int(prediction), "Unknown Career")
        else:
            predicted_career = str(prediction)
        
        # Skill gap analysis with safe access
        gap_analysis = analyze_skill_gaps(predicted_career, user_skills)
        gaps = gap_analysis.get("gaps", {}) if "error" not in gap_analysis else {}
        all_missing = []
        for level in ["essential", "intermediate", "advanced"]:
            all_missing.extend(gaps.get(level, []))
        
        resources = get_resources(all_missing)
        
        return {
            "success": True,
            "predicted_career": predicted_career,
            "confidence": f"{confidence:.1f}%",
            "prediction_class": int(prediction) if isinstance(prediction, (int, np.integer)) else str(prediction),
            "gap_analysis": gap_analysis,
            "missing_skills": all_missing,
            "learning_path": resources,
            "has_gaps": len(all_missing) > 0
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"success": False, "error": f"Pipeline failed: {str(e)}"}

if __name__ == "__main__":
    # Example raw user data - validated
    example_user = {
        'age': 20,
        'gender': 'Male',
        'tenth_percentage': 85.0,
        'twelfth_or_diploma_percentage': 82.0,
        'math_score': 8,
        'programming_score': 7,
        'communication_score': 6,
        'logical_reasoning_score': 8,
        'creativity_score': 5,
        'leadership_score': 7,
        'teamwork_score': 7,
        'problem_solving_score': 8,
        'coding_interest': 1,
        'design_interest': 0,
        'data_interest': 1,
        'business_interest': 0,
        'security_interest': 0,
        'cloud_interest': 0,
        'preferred_work_type': 'Technical',
        'preferred_domain': 'AI-ML',
        'personality_type': 'Logical',
        'certifications_count': 2,
        'project_count': 3,
        'internship_experience': 'Yes',
        'favorite_subject': 'Python'
    }
    
    user_skills = ["Python", "Excel", "SQL"]
    
    result = full_recommendation_pipeline(example_user, user_skills)
    
    if result["success"]:
        print("=== CAREER RECOMMENDATION ===")
        print(f"Recommended Career: {result['predicted_career']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Skill Gap: {result['gap_analysis'].get('gap_percentage', 'N/A')}%")
        print(f"Has Gaps: {result['has_gaps']}")
        
        print("\nMissing Skills:")
        gaps = result['gap_analysis'].get('gaps', {})
        for level, skills in gaps.items():
            if skills:
                print(f"  {level.title()}: {', '.join(skills)}")
        
        print("\nLearning Resources:")
        for rec in result['learning_path']:
            print(f"  - {rec['skill']}: {rec['resources'][0]}")
    else:
        print(f"Error: {result['error']}")

    logger.info("Pipeline execution complete")
