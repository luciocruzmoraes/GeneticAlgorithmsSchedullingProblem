import os
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def simulate_teacher_choice(
    classes_file: str = "data/classes_data.csv",
    teachers_file: str = "data/teachers_data.csv",
    output_file: str = "data/teacher_assignments.csv"
):
    """Simulate the assignment of teachers to classes based on subject expertise."""
    try:
        df_classes = pd.read_csv(classes_file)
        df_teachers = pd.read_csv(teachers_file)

        assignments = []

        for _, class_row in df_classes.iterrows():
            subject = class_row["subject"]
            possible_teachers = df_teachers[df_teachers["main_subject"] == subject]

            if possible_teachers.empty:
                # fallback: no teacher specializes in this subject
                teacher = df_teachers.sample(1).iloc[0]
                properly_assigned = "N"
            else:
                teacher = possible_teachers.sample(1).iloc[0]
                properly_assigned = "Y"

            assignments.append({
                "teacher_id": teacher["teacher_id"],
                "teacher_name": teacher["name"],
                "main_subject": teacher["main_subject"],
                "class_id": class_row["class_id"],
                "class_name": class_row["class_name"],
                "grade": class_row.get("education_level", "N/A"),
                "subject": subject,
                "num_students": class_row["num_students"],
                "lessons_per_week": class_row["lessons_per_week"],
                "properly_assigned": properly_assigned
            })

        df_assignments = pd.DataFrame(assignments)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_assignments.to_csv(output_file, index=False, encoding="utf-8")

        print(f"Teacher assignment simulation completed. File saved as '{output_file}'")
        return df_assignments

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return None
    except KeyError as e:
        print(f"Error: Missing required column in data files: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during teacher assignment simulation: {e}")
        return None


if __name__ == "__main__":
    simulate_teacher_choice()
