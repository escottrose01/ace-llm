def main(
    name: str, dob: str, city: str, email: str, preferred_date: str, medical_issues: str, preferred_doctor: str = None
) -> str:
    """
    Book a healthcare appointment and return the details.

    :param name: str: The name of the patient.
    :param dob: str: The date of birth of the patient.
    :param city: str: The city where the patient resides.
    :param email: str: The email address of the patient.
    :param preferred_date: str: The preferred date for the appointment.
    :param medical_issues: str: Description of the patient's medical issues.
    :param preferred_doctor: str: The preferred doctor for the appointment, if any.
    :returns: dict: The details of the appointment.
    """

    import json
    import random
    from datetime import time

    hours = random.randint(8, 17)  # Assuming office hours are 8am to 5pm
    # Appointment at 0, 15, 30, or 45 mins past the hour
    minutes = random.choice([0, 15, 30, 45])
    appointment_time = time(hour=hours, minute=minutes)

    appointment_details = {
        "Patient Name": name,
        "Date of Birth": dob,
        "City": city,
        "Email Address": email,
        "Appointment Date and Time": preferred_date + " " + str(appointment_time),
        "Medical Issues": medical_issues,
        "Preferred Doctor": preferred_doctor if preferred_doctor else "Any available",
    }

    # return appointment_details

    # Deviating from SecGPT to force a return type of str
    return json.dumps(appointment_details, indent=4)
