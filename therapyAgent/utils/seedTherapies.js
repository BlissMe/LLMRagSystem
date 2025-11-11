

const defaultTherapies = [
    {
        therapyID: "T001",
        name: "Anxiety_Games",
        applicableLevel: "Moderate",
        description:
            "A fun game-based therapy to reduce anxiety levels through interactive relaxation challenges.",
        path: "/therapy/Game/Anxiety_Games.tsx",
        durationMinutes: 15,
    },
    {
        therapyID: "T002",
        name: "breathing-game",
        applicableLevel: "Moderate",
        description:
            "A breathing control game designed to synchronize breathing patterns with calming visuals.",
        path: "/therapy/breathing-game.tsx",
        durationMinutes: 10,
    },
    {
        therapyID: "T003",
        name: "BreathingExercise",
        applicableLevel: "Minimal",
        description:
            "Simple guided breathing exercises to ease mild anxiety and enhance focus.",
        path: "/therapy/BreathingExercise.tsx",
        durationMinutes: 8,
    },
    {
        therapyID: "T004",
        name: "forest-game",
        applicableLevel: "Moderate",
        description:
            "Nature-themed mindfulness therapy to immerse the user in calming virtual forest experiences.",
        path: "/therapy/forest-game.tsx",
        durationMinutes: 12,
    },
    {
        therapyID: "T005",
        name: "LogMood",
        applicableLevel: "Minimal",
        description:
            "A daily mood logging therapy that helps users track emotions and patterns for better awareness.",
        path: "/therapy/LogMood.tsx",
        durationMinutes: 5,
    },
    {
        therapyID: "T006",
        name: "MeditationPlayer",
        applicableLevel: "Moderate",
        description:
            "A meditation player that provides guided sessions for stress management and inner peace.",
        path: "/therapy/MeditationPlayer.tsx",
        durationMinutes: 15,
    },
    {
        therapyID: "T007",
        name: "MoodTracker",
        applicableLevel: "Minimal",
        description:
            "An interactive mood tracking module to monitor emotional states over time.",
        path: "/therapy/MoodTracker.tsx",
        durationMinutes: 7,
    },
    {
        therapyID: "T008",
        name: "MoodTrackerMain",
        applicableLevel: "Minimal",
        description:
            "Main dashboard for mood tracking and visualization of emotional trends.",
        path: "/therapy/MoodTrackerMain.tsx",
        durationMinutes: 10,
    },
    {
        therapyID: "T009",
        name: "ocean-waves",
        applicableLevel: "Moderate",
        description:
            "Audio-visual therapy simulating ocean waves for deep relaxation and mindfulness.",
        path: "/therapy/ocean-waves.tsx",
        durationMinutes: 10,
    },
    {
        therapyID: "T010",
        name: "zen-garden",
        applicableLevel: "Severe",
        description:
            "A virtual Zen garden experience for reflection, focus, and cognitive grounding.",
        path: "/therapy/zen-garden.tsx",
        durationMinutes: 20,
    },
];

async function seedTherapies() {
    try {
        await mongoose.connect(MONGO_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });

        console.log(" Connected to MongoDB");

        // Clear existing therapies before seeding
        await Therapy.deleteMany({});
        console.log(" Cleared existing therapies");

        // Insert new ones
        await Therapy.insertMany(defaultTherapies);
        console.log(" Default therapies inserted successfully!");

        process.exit(0);
    } catch (error) {
        console.error(" Error seeding therapies:", error);
        process.exit(1);
    }
}

seedTherapies();
