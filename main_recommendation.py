import pandas as pd
import joblib

pipeline = joblib.load("career_pipeline.pkl")  # or career_model.pkl if that's your good one

# YOUR real profile - Diploma student from Gurugram
your_profile = pd.DataFrame([{
    "current_stage": "Pursuing_Diploma",
    "current_course_or_class": "Diploma_CSE",  
    "age": 20,
    "gender": "Male",
    "tenth_percentage": 82,
    "twelfth_or_diploma_percentage": 78,
    "graduation_percentage": 0,
    "math_score": 8,
    "programming_score": 9,
    "communication_score": 7,
    "logical_reasoning_score": 8,
    "creativity_score": 6,
    "leadership_score": 6,
    "teamwork_score": 8,
    "problem_solving_score": 9,
    "coding_interest": 9,
    "design_interest": 7,
    "data_interest": 9,
    "business_interest": 8,
    "security_interest": 5,
    "cloud_interest": 8,
    "preferred_work_type": "Technical",
    "preferred_subroles": "Data Analysis, Web Development",
    "preferred_domain": "Data",
    "personality_type": "Logical",
    "certifications_count": 2,
    "project_count": 3,
    "internship_experience": "No",
    "favorite_subject": "Math",
    "current_skills": "Python, SQL, HTML, CSS, Pandas, PowerBI"
}])

prediction = pipeline.predict(your_profile)[0]
probabilities = pipeline.predict_proba(your_profile)[0]

print("Your Predicted Career:", prediction)
print("Top 3 probabilities:")
top3 = sorted(zip(pipeline.classes_, probabilities), reverse=True)[:3]
for career, prob in top3:
    print(f"  {career}: {prob:.1%}")