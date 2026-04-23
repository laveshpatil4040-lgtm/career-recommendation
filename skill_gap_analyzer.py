from requirnments import CAREER_REQUIREMENTS

def analyze_skill_gaps(predicted_career, user_skills):
    if predicted_career not in CAREER_REQUIREMENTS:
        return {
            "predicted_career": predicted_career,
            "gaps": {
                "essential": [],
                "intermediate": [],
                "advanced": []
            },
            "gap_percentage": 0,
            "has_gaps": False,
            "error": f"No requirement matrix found for {predicted_career}"
        }

    requirements = CAREER_REQUIREMENTS[predicted_career]
    user_skills_lower = [skill.strip().lower() for skill in user_skills]

    gaps = {
        "essential": [skill for skill in requirements["essential"] if skill.strip().lower() not in user_skills_lower],
        "intermediate": [skill for skill in requirements["intermediate"] if skill.strip().lower() not in user_skills_lower],
        "advanced": [skill for skill in requirements["advanced"] if skill.strip().lower() not in user_skills_lower]
    }

    total_required = sum(len(skills) for skills in requirements.values())
    total_gaps = sum(len(gaps[level]) for level in gaps)
    gap_percentage = (total_gaps / total_required) * 100 if total_required > 0 else 0

    return {
        "predicted_career": predicted_career,
        "gaps": gaps,
        "gap_percentage": round(gap_percentage, 1),
        "has_gaps": total_gaps > 0
    }