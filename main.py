from fastapi import FastAPI

print("Starting minimal app...", flush=True)

app = FastAPI()

@app.get("/")
async def root():
    print("Root endpoint called", flush=True)
    return {"status": "ok"}
