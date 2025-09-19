from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import random

router = APIRouter()

# Game state variables
secret_number = 0
game_active = False

class GuessRequest(BaseModel):
    guess: int

@router.post("/start")
def start_game():
    """Starts a new game by generating a new secret number."""
    global secret_number, game_active
    secret_number = random.randint(1, 100)
    game_active = True
    print(f"New game started. The secret number is {secret_number}")  # For debugging
    return {"message": "New game started! Guess a number between 1 and 100."}

@router.post("/guess")
def handle_guess(data: GuessRequest):
    """Handles a user's guess and returns feedback."""
    global game_active, secret_number

    if not game_active:
        raise HTTPException(status_code=400, detail="Game has not been started yet.")

    user_guess = data.guess

    if user_guess < secret_number:
        return {"message": "📈 Too low! Try again."}
    elif user_guess > secret_number:
        return {"message": "📉 Too high! Try again."}
    else:
        game_active = False
        return {"message": f"🎉 Correct! The number was {secret_number}."}
