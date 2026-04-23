import pandas as pd
import numpy as np
import random

df = pd.read_csv("career_recommendation_v3_industry.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Career labels:", df["career_label"].value_counts())

def clip(val, low, high):
    return max(low, min(high, val))

def generate_new_row(row, new_id):
    new_row = row.copy()

    new_row["student_id"] = new_id
    new_row["age"] = clip(int(row["age"] + random.choice([-1, 0, 1])), 17, 25)

    score_cols = [
        "math_score", "programming_score", "communication_score",
        "logical_reasoning_score", "creativity_score",
        "leadership_score", "teamwork_score", "problem_solving_score"
    ]

    for col in score_cols:
        new_row[col] = clip(int(row[col] + random.randint(-8, 8)), 35, 100)

    percent_cols = ["tenth_percentage", "twelfth_or_diploma_percentage"]
    for col in percent_cols:
        new_row[col] = round(clip(row[col] + random.uniform(-5, 5), 50, 99), 2)

    interest_cols = [
        "coding_interest", "design_interest", "data_interest",
        "business_interest", "security_interest", "cloud_interest"
    ]

    for col in interest_cols:
        new_row[col] = clip(int(row[col] + random.randint(-2, 2)), 1, 10)

    new_row["certifications_count"] = clip(int(row["certifications_count"] + random.choice([-1, 0, 1])), 0, 7)
    new_row["project_count"] = clip(int(row["project_count"] + random.choice([-1, 0, 1, 2])), 0, 10)

    return new_row

def apply_career_logic(new_row):
    career = new_row["career_label"]

    if career in ["Junior ML Engineer", "AI/ML Engineer Trainee", "Junior Data Scientist"]:
        new_row["math_score"] = clip(new_row["math_score"] + random.randint(3, 8), 35, 100)
        new_row["programming_score"] = clip(new_row["programming_score"] + random.randint(3, 8), 35, 100)
        new_row["logical_reasoning_score"] = clip(new_row["logical_reasoning_score"] + random.randint(3, 8), 35, 100)
        new_row["preferred_domain"] = "AI-ML"

    elif career in ["Frontend Developer", "UI/UX Designer"]:
        new_row["creativity_score"] = clip(new_row["creativity_score"] + random.randint(4, 10), 35, 100)
        new_row["design_interest"] = clip(new_row["design_interest"] + random.randint(2, 4), 1, 10)
        new_row["preferred_domain"] = "Web"

    elif career in ["Backend Developer", "Full Stack Developer", "Junior Web Developer"]:
        new_row["coding_interest"] = clip(new_row["coding_interest"] + random.randint(2, 4), 1, 10)
        new_row["programming_score"] = clip(new_row["programming_score"] + random.randint(3, 8), 35, 100)
        new_row["preferred_domain"] = "Web"

    elif career in ["Associate Data Analyst", "Junior Business Analyst", "Database Administrator"]:
        new_row["data_interest"] = clip(new_row["data_interest"] + random.randint(2, 4), 1, 10)

    elif career in ["Security Analyst Trainee"]:
        new_row["security_interest"] = clip(new_row["security_interest"] + random.randint(3, 5), 1, 10)
        new_row["preferred_domain"] = "Security"

    elif career in ["Junior DevOps Engineer", "Cloud Operations Engineer"]:
        new_row["cloud_interest"] = clip(new_row["cloud_interest"] + random.randint(3, 5), 1, 10)
        new_row["preferred_domain"] = "Cloud"

    return new_row

new_rows = []
start_num = 2000

for i in range(2000):   # change this number to create more rows
    sample_row = df.sample(1).iloc[0]
    new_id = f"S{start_num + i:04d}"
    new_row = generate_new_row(sample_row, new_id)
    new_row = apply_career_logic(new_row)
    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)
expanded_df = pd.concat([df, new_df], ignore_index=True)

print("Final rows:", len(expanded_df))
print("Missing values:\n", expanded_df.isnull().sum())
print("Duplicate student_id:", expanded_df["student_id"].duplicated().sum())
print("Duplicate rows:", expanded_df.duplicated().sum())
print("Career distribution:\n", expanded_df["career_label"].value_counts())

expanded_df = expanded_df.drop_duplicates()
expanded_df = expanded_df.drop_duplicates(subset=["student_id"])

expanded_df.to_csv("career_recommendation_expanded.csv", index=False)
print("Expanded dataset saved successfully.")

print("Original rows:", len(df))
print("New generated rows:", len(new_df))
print("Final rows:", len(expanded_df))
print("File saved successfully as career_recommendation_expanded.csv")

import os
print("Current folder:", os.getcwd())