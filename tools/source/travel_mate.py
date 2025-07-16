def main(
    name: str = "",
    email: str = "",
    departure_city: str = "",
    destination_city: str = "",
    departure_date: str = "",
    class_of_service: str = "",
    special_requirements: str = "",
) -> str:
    import json
    import random
    import string
    import time

    booking_reference = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    hours = random.randint(8, 17)  # Assuming office hours are 8am to 5pm
    # Appointment at 0, 15, 30, or 45 mins past the hour
    minutes = random.choice([0, 15, 30, 45])
    appointment_time = time(hour=hours, minute=minutes)

    booking_details = {
        "Passenger Name": name,
        "Email Address": email,
        "Departure City": departure_city,
        "Destination City": destination_city,
        "Departure Date and Time": departure_date + " " + str(appointment_time),
        "Class of Service": class_of_service,
        "Special Requirements": special_requirements,
        "Booking Reference": booking_reference,
    }

    # return booking_details

    # Deviating from SecGPT to force a return type of str
    return {"result": json.dumps(booking_details, indent=4)}
