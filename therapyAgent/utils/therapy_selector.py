import random

default_therapies = [
    {
        "therapyID": "T001",
        "name": "Listen Me",
        "applicableLevel": "Moderate",
        "description": "Provide Relaxing musics to calm the mind.",
        "path": "/dash/anxiety",
        "durationMinutes": 6,
    },
    {
        "therapyID": "T002",
        "name": "Mindful Breathing",
        "applicableLevel": "Moderate",
        "description": "A breathing control game designed to synchronize breathing patterns with calming visuals.",
        "path": "/therapy/breathing",
        "durationMinutes": 10,
    },
    {
        "therapyID": "T003",
        "name": "Meditation Session",
        "applicableLevel": "Minimal",
        "description": "Simple guided to hearing raining or sunny day sound to feeling relaxed.",
        "path": "/therapy/medication",
        "durationMinutes": 8,
    },
    {
        "therapyID": "T004",
        "name": "Mindful Forest",
        "applicableLevel": "Moderate",
        "description": "Nature-themed mindfulness therapy to immerse the user in calming virtual forest experiences.",
        "path": "/dash/forest",
        "durationMinutes": 12,
    },
    {
        "therapyID": "T005",
        "name": "LogMood",
        "applicableLevel": "Minimal",
        "description": "A daily mood logging therapy that helps users track emotions and patterns for better awareness.",
        "path": "/therapy/mood-tracker-home",
        "durationMinutes": 5,
    },
    {
        "therapyID": "T006",
        "name": "Body Scan Therapy",
        "applicableLevel": "Moderate",
        "description": "Guided body scan meditation to promote relaxation and body awareness.",
        "path": "/therapy/body-scan",
        "durationMinutes": 15,
    },
    {
        "therapyID": "T007",
        "name": "MoodTracker",
        "applicableLevel": "Minimal",
        "description": "An interactive mood tracking module to monitor emotional states over time.",
        "path": "/therapy/mood-tracker-home",
        "durationMinutes": 7,
    },
    {
        "therapyID": "T008",
        "name": "Number Guessing Game",
        "applicableLevel": "Minimal",
        "description": "A fun and engaging number guessing game to distract and entertain users.",
        "path": "/game/therapy_game",
        "durationMinutes": 10,
    },
    {
        "therapyID": "T009",
        "name": "ocean-waves",
        "applicableLevel": "Moderate",
        "description": "Audio-visual therapy simulating ocean waves for deep relaxation and mindfulness.",
        "path": "/dash/ocean",
        "durationMinutes": 10,
    },
    {
        "therapyID": "T010",
        "name": "Zen Garden",
        "applicableLevel": "Moderate",
        "description": "A virtual Zen garden experience for reflection, focus, and cognitive grounding.",
        "path": "/dash/zen",
        "durationMinutes": 5,
    },
     {
        "therapyID": "T0011",
        "name": "Anxiety_Games",
        "applicableLevel": "Moderate",
        "description": "A fun game-based therapy to reduce anxiety levels through interactive relaxation challenges.",
        "path": "/dash/anxiety",
        "durationMinutes": 15,
    },
]

def get_therapy_recommendation(db, depression_level, history_records):
    """
    Select therapy from default list based on depression level and past usage.
    """
    # Filter by applicable level
    matching = [
        t for t in default_therapies
        if t["applicableLevel"].lower() in [depression_level.lower(), "general"]
    ]

    # If none match, fallback to all
    if not matching:
        matching = default_therapies

    # Avoid recently used therapies
    used_ids = [h["therapy_id"] for h in history_records]
    available = [t for t in matching if t["therapyID"] not in used_ids]

    # Pick a random one
    selected = random.choice(available or matching)

    return {
        "id": selected["therapyID"],
        "name": selected["name"],
        "description": selected["description"],
        "path": selected["path"]
    }
