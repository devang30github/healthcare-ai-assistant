from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Mock availability data — simulates a real scheduling system
MOCK_SLOTS = {
    "cardiology": {
        "monday":    ["9:00 AM", "11:00 AM", "3:00 PM"],
        "tuesday":   ["10:00 AM", "2:00 PM"],
        "wednesday": ["9:00 AM", "1:00 PM", "4:00 PM"],
        "thursday":  ["11:00 AM", "3:00 PM"],
        "friday":    ["9:00 AM", "10:00 AM"],
        "saturday":  ["10:00 AM"],
    },
    "general": {
        "monday":    ["8:00 AM", "9:00 AM", "10:00 AM", "2:00 PM", "4:00 PM"],
        "tuesday":   ["8:00 AM", "11:00 AM", "3:00 PM"],
        "wednesday": ["9:00 AM", "10:00 AM", "1:00 PM"],
        "thursday":  ["8:00 AM", "2:00 PM", "4:00 PM"],
        "friday":    ["9:00 AM", "11:00 AM", "3:00 PM"],
        "saturday":  ["10:00 AM", "11:00 AM"],
    },
    "dermatology": {
        "monday":    ["10:00 AM", "2:00 PM"],
        "tuesday":   ["9:00 AM", "3:00 PM"],
        "wednesday": ["11:00 AM"],
        "thursday":  ["10:00 AM", "2:00 PM"],
        "friday":    ["9:00 AM"],
        "saturday":  [],
    },
    "telehealth": {
        "monday":    ["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "6:00 PM"],
        "tuesday":   ["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM"],
        "wednesday": ["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "6:00 PM"],
        "thursday":  ["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM"],
        "friday":    ["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM"],
        "saturday":  ["10:00 AM", "12:00 PM", "2:00 PM"],
    },
}

SUPPORTED_DEPARTMENTS = list(MOCK_SLOTS.keys())


def check_available_slots(department: str, day: str) -> str:
    """
    Mock appointment slot checker.
    Returns available time slots for a given department and day.

    Args:
        department: e.g. 'cardiology', 'general', 'dermatology', 'telehealth'
        day:        e.g. 'monday', 'tuesday', ...

    Returns:
        A human-readable string listing available slots.
    """
    dept = department.lower().strip()
    day  = day.lower().strip()

    logger.info(f"Checking slots | department={dept} | day={day}")

    # Validate department
    if dept not in MOCK_SLOTS:
        available = ", ".join(SUPPORTED_DEPARTMENTS)
        return (
            f"Sorry, I don't have scheduling information for '{department}'. "
            f"Available departments are: {available}."
        )

    # Validate day
    if day not in MOCK_SLOTS[dept]:
        return (
            f"Sorry, '{day}' is not a valid day. "
            f"Please specify a day from Monday to Saturday."
        )

    slots = MOCK_SLOTS[dept][day]

    if not slots:
        return (
            f"There are no available slots for {department.title()} on {day.title()}. "
            f"Please try a different day."
        )

    slot_list = ", ".join(slots)
    return (
        f"Available slots for {department.title()} on {day.title()}: {slot_list}. "
        f"To confirm a booking, please contact our scheduling desk or use the patient portal."
    )