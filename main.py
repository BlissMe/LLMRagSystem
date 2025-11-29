from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 


from textChatMode.chat import router as ask_router
#from textChatMode.chatmistral import router as ask_router
from voiceChatMode.voice import router as voice_router
#from face_recognition_auth.auth_face_recognition import router as face_router
from LevelDetection.router.levelDetection import router as level_detection_router
#from textChatMode.assesmentAgent.routes import router as agent_router
from CountingGame.game import router as game_router
from monitoringAgentSystem.monotoring_agent_system import router as monitoring_router
#from MonitoringAgent.monitor_api import router as monitor_router
from LearningAgent.Feedback_Therapy.app_fastapi import router as therapyFeedback_router
from therapyAgent.therapyAgent import router as therapy_router

app = FastAPI()
 
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes from other files
app.include_router(ask_router)
app.include_router(voice_router)
#app.include_router(face_router)
app.include_router(level_detection_router)
#app.include_router(agent_router)
app.include_router(game_router)
app.include_router(therapy_router)
app.include_router(monitoring_router)
#app.include_router(monitor_router)
app.include_router(therapy_router)
app.include_router(therapyFeedback_router)

@app.get("/")
def root():
    return {"message": "All endpoints are loaded successfully"}
