from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import pickle
import joblib
import sqlite3
import os
import re
import json
import csv
import io
from datetime import datetime

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "career_recommendation_secret_key_2026")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "career_app.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
MODEL_FALLBACK_PATH = os.path.join(BASE_DIR, "career_model.pkl")
PIPELINE_FALLBACK_PATH = os.path.join(BASE_DIR, "career_pipeline.pkl")

model = None
label_encoder = None
openai_client = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            predicted_career TEXT NOT NULL,
            confidence TEXT NOT NULL,
            skills_needed_json TEXT NOT NULL,
            top3_json TEXT NOT NULL,
            input_snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    seed_users = {"admin": "admin123", "student": "student123"}
    for username, password in seed_users.items():
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), datetime.utcnow().isoformat()),
            )
    conn.commit()
    conn.close()


def load_serialized(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


def init_model():
    global model, label_encoder
    try:
        model = load_serialized(MODEL_PATH)
        print("Career model loaded successfully from:", MODEL_PATH)
    except Exception as e:
        print("Primary model load failed:", e)
    if model is None:
        for fallback_path in [MODEL_FALLBACK_PATH, PIPELINE_FALLBACK_PATH]:
            try:
                if os.path.exists(fallback_path):
                    model = load_serialized(fallback_path)
                    print("Career model loaded successfully from:", fallback_path)
                    break
            except Exception as e:
                print(f"Error loading fallback model at {fallback_path}:", e)
    try:
        if os.path.exists(ENCODER_PATH):
            label_encoder = load_serialized(ENCODER_PATH)
            print("Label encoder loaded successfully from:", ENCODER_PATH)
    except Exception as e:
        print("Error loading label_encoder.pkl:", e)


def init_openai():
    global openai_client
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if OpenAI and api_key:
        try:
            openai_client = OpenAI(api_key=api_key)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print("OpenAI client initialization failed:", e)


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def sanitize_text(value, default):
    text = str(value).strip() if value is not None else ""
    return text or default


def is_valid_username(username):
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username))


def build_input_row(data):
    return {
        "current_stage": sanitize_text(data.get("current_stage"), "Pursuing_Diploma"),
        "current_course_or_class": sanitize_text(data.get("current_course_or_class"), "Diploma_CSE"),
        "age": clamp(safe_int(data.get("age"), 18), 15, 35),
        "gender": sanitize_text(data.get("gender"), "Male"),
        "tenth_percentage": clamp(safe_float(data.get("tenth_percentage"), 70), 0, 100),
        "twelfth_or_diploma_percentage": clamp(safe_float(data.get("twelfth_or_diploma_percentage"), 0), 0, 100),
        "graduation_percentage": clamp(safe_float(data.get("graduation_percentage"), 0), 0, 100),
        "math_score": clamp(safe_int(data.get("math_score"), 5), 1, 10),
        "programming_score": clamp(safe_int(data.get("programming_score"), 5), 1, 10),
        "communication_score": clamp(safe_int(data.get("communication_score"), 5), 1, 10),
        "logical_reasoning_score": clamp(safe_int(data.get("logical_reasoning_score"), 5), 1, 10),
        "creativity_score": clamp(safe_int(data.get("creativity_score"), 5), 1, 10),
        "leadership_score": clamp(safe_int(data.get("leadership_score"), 5), 1, 10),
        "teamwork_score": clamp(safe_int(data.get("teamwork_score"), 5), 1, 10),
        "problem_solving_score": clamp(safe_int(data.get("problem_solving_score"), 5), 1, 10),
        "coding_interest": clamp(safe_int(data.get("coding_interest"), 5), 1, 10),
        "design_interest": clamp(safe_int(data.get("design_interest"), 5), 1, 10),
        "business_interest": clamp(safe_int(data.get("business_interest"), 5), 1, 10),
        "healthcare_interest": clamp(safe_int(data.get("healthcare_interest"), 1), 1, 10),
        "research_interest": clamp(safe_int(data.get("research_interest"), 5), 1, 10),
        "management_interest": clamp(safe_int(data.get("management_interest"), 5), 1, 10),
        "finance_interest": clamp(safe_int(data.get("finance_interest"), 3), 1, 10),
        "education_interest": clamp(safe_int(data.get("education_interest"), 3), 1, 10),
        "cloud_interest": clamp(safe_int(data.get("cloud_interest"), 5), 1, 10),
        "security_interest": clamp(safe_int(data.get("security_interest"), 3), 1, 10),
        "data_interest": clamp(safe_int(data.get("data_interest"), 7), 1, 10),
        "preferred_work_type": sanitize_text(data.get("preferred_work_type"), "Software Development"),
        "preferred_subroles": sanitize_text(data.get("preferred_subroles"), "Web Developer"),
        "preferred_domain": sanitize_text(data.get("preferred_domain"), "IT"),
        "personality_type": sanitize_text(data.get("personality_type"), "Ambivert"),
        "certifications_count": clamp(safe_int(data.get("certifications_count"), 0), 0, 20),
        "project_count": clamp(safe_int(data.get("project_count"), 0), 0, 20),
        "internship_experience": clamp(safe_int(data.get("internship_experience"), 0), 0, 20),
        "favorite_subject": sanitize_text(data.get("favorite_subject"), "Programming"),
        "current_skills": sanitize_text(data.get("current_skills"), "Basic Computer Knowledge"),
    }


def prepare_dataframe(input_row):
    input_df = pd.DataFrame([input_row])
    if model is not None and hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(columns=list(model.feature_names_in_), fill_value=0)
    return input_df


def get_top_predictions(probabilities, top_n=3):
    top_idx = np.argsort(probabilities)[::-1][:top_n]
    results = []
    for i in top_idx:
        career_name = str(i)
        if model is not None and hasattr(model, "classes_"):
            try:
                career_name = str(model.classes_[i])
            except Exception:
                pass
        elif label_encoder is not None:
            try:
                career_name = str(label_encoder.inverse_transform([i])[0])
            except Exception:
                pass
        results.append({"career": career_name, "score": f"{probabilities[i] * 100:.2f}%"})
    return results


def get_skill_recommendations(input_row):
    skills_needed = []
    if input_row["programming_score"] <= 3:
        skills_needed.append("Python")
    if input_row["communication_score"] <= 3:
        skills_needed.append("Communication")
    if input_row["logical_reasoning_score"] <= 3:
        skills_needed.append("Logical Reasoning")
    if input_row["problem_solving_score"] <= 3:
        skills_needed.append("Problem Solving")
    if input_row["coding_interest"] >= 7:
        skills_needed.append("JavaScript")
    if input_row["data_interest"] >= 7:
        skills_needed.append("SQL")
    if input_row["cloud_interest"] >= 7:
        skills_needed.append("Cloud")
    if input_row["security_interest"] >= 7:
        skills_needed.append("Cyber Security")
    if input_row["project_count"] == 0:
        skills_needed.append("Projects")
    if input_row["internship_experience"] == 0:
        skills_needed.append("Internships")
    return list(dict.fromkeys(skills_needed))


def calculate_readiness_score(input_row):
    score_components = {
        "academics": (
            input_row["tenth_percentage"] * 0.4
            + input_row["twelfth_or_diploma_percentage"] * 0.4
            + input_row["graduation_percentage"] * 0.2
        ),
        "core_skills": (
            input_row["programming_score"]
            + input_row["logical_reasoning_score"]
            + input_row["problem_solving_score"]
        )
        / 3
        * 10,
        "communication": input_row["communication_score"] * 10,
        "practical_experience": min(
            (input_row["project_count"] * 15) + (input_row["internship_experience"] * 20) + (input_row["certifications_count"] * 8),
            100,
        ),
    }

    weighted = (
        score_components["academics"] * 0.30
        + score_components["core_skills"] * 0.35
        + score_components["communication"] * 0.15
        + score_components["practical_experience"] * 0.20
    )
    total_score = round(max(0, min(weighted, 100)), 1)
    return total_score, {k: round(v, 1) for k, v in score_components.items()}


def build_learning_roadmap(predicted_career, skills_needed):
    normalized = [s.lower() for s in (skills_needed or [])]
    phase1 = [
        "Finalize one target role and update your weekly study plan",
        "Strengthen communication and problem solving basics",
    ]
    phase2 = [
        "Build 2 projects aligned to your target career",
        "Publish project code and documentation on GitHub",
    ]
    phase3 = [
        "Prepare internship resume and LinkedIn portfolio",
        "Practice interviews and apply consistently every week",
    ]

    if "python" in normalized:
        phase1.append("Complete Python fundamentals and solve beginner problems")
    if "sql" in normalized:
        phase1.append("Practice SQL queries and mini database tasks")
    if "javascript" in normalized:
        phase1.append("Learn JavaScript DOM, events, and API usage")
    if "projects" in normalized:
        phase2.append("Start with one mini project before major project")
    if "internships" in normalized:
        phase3.append("Apply to 5-10 internships weekly with tailored resumes")

    return {
        "target_career": predicted_career,
        "phase_1_foundation_0_to_30_days": phase1,
        "phase_2_projects_31_to_60_days": phase2,
        "phase_3_placement_61_to_90_days": phase3,
    }


def save_prediction(username, payload, input_row):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions
        (username, predicted_career, confidence, skills_needed_json, top3_json, input_snapshot_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            payload["career"],
            payload["confidence"],
            json.dumps(payload["skills_needed"]),
            json.dumps(payload["top3"]),
            json.dumps(input_row),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user_name=session["user"])


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip() or request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_connection()
        user = conn.execute("SELECT username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["username"]
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        if username == "" or password == "":
            return render_template("signup.html", error="Please fill all fields.")
        if not is_valid_username(username):
            return render_template("signup.html", error="Username must be 3-32 chars and use letters, numbers, _, -, or .")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")
        conn = get_db_connection()
        exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            conn.close()
            return render_template("signup.html", error="Username already exists.")
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "user" not in session:
            return jsonify({"error": "Unauthorized access. Please login first."}), 401
        if model is None:
            return jsonify({"error": f"Model file not loaded. Put model.pkl inside: {BASE_DIR}"}), 500
        if not request.is_json:
            return jsonify({"error": "Invalid request. JSON body is required."}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload."}), 400
        input_row = build_input_row(data)
        input_df = prepare_dataframe(input_row)
        prediction = model.predict(input_df)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = f"{np.max(probabilities) * 100:.2f}%"
            top3 = get_top_predictions(probabilities, top_n=3)
        else:
            confidence = "Prediction generated"
            top3 = []
        predicted_career = str(prediction)
        if label_encoder is not None and isinstance(prediction, (int, np.integer)):
            try:
                predicted_career = str(label_encoder.inverse_transform([prediction])[0])
            except Exception:
                pass
        if not top3:
            top3 = [{"career": predicted_career, "score": confidence}]
        payload = {
            "career": predicted_career,
            "confidence": confidence,
            "top3": top3,
            "skills_needed": get_skill_recommendations(input_row),
        }
        save_prediction(session["user"], payload, input_row)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT predicted_career, confidence, created_at
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        LIMIT 10
        """,
        (session["user"],),
    ).fetchall()
    conn.close()
    return jsonify({"history": [{"career": r["predicted_career"], "confidence": r["confidence"], "created_at": r["created_at"]} for r in rows]})


@app.route("/insights", methods=["GET"])
def insights():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE username = ?", (session["user"],)).fetchone()["c"]
    top = conn.execute(
        """
        SELECT predicted_career, COUNT(*) AS cnt
        FROM predictions
        WHERE username = ?
        GROUP BY predicted_career
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (session["user"],),
    ).fetchone()
    conn.close()
    return jsonify(
        {
            "total_predictions": int(total),
            "top_career": top["predicted_career"] if top else "N/A",
            "top_career_count": int(top["cnt"]) if top else 0,
        }
    )


@app.route("/career-readiness", methods=["POST"])
def career_readiness():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    if not request.is_json:
        return jsonify({"error": "Invalid request. JSON body is required."}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    input_row = build_input_row(data)
    total_score, breakdown = calculate_readiness_score(input_row)
    level = "Beginner"
    if total_score >= 75:
        level = "Advanced"
    elif total_score >= 50:
        level = "Intermediate"

    return jsonify(
        {
            "readiness_score": total_score,
            "readiness_level": level,
            "breakdown": breakdown,
        }
    )


@app.route("/learning-roadmap", methods=["POST"])
def learning_roadmap():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    if not request.is_json:
        return jsonify({"error": "Invalid request. JSON body is required."}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    predicted_career = sanitize_text(data.get("career"), "Career Path")
    skills_needed = data.get("skills_needed", [])
    if not isinstance(skills_needed, list):
        skills_needed = []

    roadmap = build_learning_roadmap(predicted_career, skills_needed)
    return jsonify(roadmap)


@app.route("/export-history", methods=["GET"])
def export_history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT predicted_career, confidence, created_at
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        """,
        (session["user"],),
    ).fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["career", "confidence", "created_at"])
    for row in rows:
        writer.writerow([row["predicted_career"], row["confidence"], row["created_at"]])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={session['user']}_career_history.csv"},
    )


@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access. Please login first."}), 401
    if not request.is_json:
        return jsonify({"error": "Invalid request. JSON body is required."}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400
    user_message = str(data.get("message", "")).strip()
    predicted_career = str(data.get("career", "")).strip()
    if not user_message:
        return jsonify({"error": "Please enter a message."}), 400

    if openai_client is None:
        lower = user_message.lower()
        if "internship" in lower:
            return jsonify({"reply": "Build 2 projects, update LinkedIn, and apply to at least 5 internships each week."})
        if "project" in lower:
            return jsonify({"reply": "Create one mini project and one major project aligned with your target role and publish both on GitHub."})
        if "skill" in lower:
            return jsonify({"reply": "Prioritize communication, problem solving, and one deep technical skill for your target career."})
        return jsonify({"reply": "Ask me about career paths, internships, projects, certifications, and learning roadmaps."})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a practical student career guidance assistant."},
                {"role": "user", "content": f"Predicted career: {predicted_career}\nQuestion: {user_message}"},
            ],
            temperature=0.7,
            max_completion_tokens=300,
        )
        return jsonify({"reply": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 400


init_database()
init_model()
init_openai()

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)