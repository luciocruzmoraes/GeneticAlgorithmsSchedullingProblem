import pandas as pd
import random
import os

def simulate_teacher_choice(
    classes_file: str = "classes_data.csv",
    teachers_file: str = "teachers_data.csv",
    output_file: str = "teacher_assignments.csv"
):
    """
    Simulates teacher assignments for each subject within each class.

    It ensures that each subject is assigned to a teacher who lists it as
    their main subject. If no suitable teacher is found, it assigns a random
    teacher as a fallback.

    The results are saved to the specified output file.
    """
    try:
        # Load classes and teachers data
        df_classes = pd.read_csv(classes_file)
        df_teachers = pd.read_csv(teachers_file)

        assignments = []
        for _, class_row in df_classes.iterrows():
            subject = class_row["subject"]

            # Filter teachers by their main subject
            possible_teachers = df_teachers[df_teachers["main_subject"] == subject]

            if possible_teachers.empty:
                # Fallback: if no teacher has the specialty, assign a random one
                teacher = df_teachers.sample(1).iloc[0]
                was_properly_assigned = "N"  # Flag this as a forced assignment
            else:
                # Assign a random teacher from the qualified pool
                teacher = possible_teachers.sample(1).iloc[0]
                was_properly_assigned = "Y"

            assignments.append({
                "teacher_id": teacher["teacher_id"],
                "teacher_name": teacher["name"],
                "main_subject": teacher["main_subject"],
                "class_id": class_row["class_id"],
                "class_name": class_row["class_name"],
                "grade": class_row["grade"],
                "subject": subject,
                "num_students": class_row["num_students"],
                "lessons_per_week": class_row["lessons_per_week"],
                "was_properly_assigned": was_properly_assigned
            })

        # Save the assignments to a CSV file
        df_assignments = pd.DataFrame(assignments)
        df_assignments.to_csv(output_file, index=False, encoding="utf-8")

        print(f"Teacher assignment simulation completed. File saved as '{output_file}'")
        return df_assignments

    except FileNotFoundError as e:
        print(f"Error: The file {e.filename} was not found.")
        return None
    except KeyError as e:
        print(f"Error: A required column was not found in the data files: {e}. Please check your CSVs.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during teacher assignment: {e}")
        return None


if __name__ == "__main__":
    simulate_teacher_choice()