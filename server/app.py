from fastapi import FastAPI
import uvicorn
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()
env = MedicalTriageEnvironment(task_id="triage_basic")

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump() if obs else None

@app.post("/step")
def step(action: MedicalTriageAction):
    result = env.step(action)
    return {
        "observation": result.observation.model_dump() if result.observation else None,
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.get("/state")
def state():
    return env.state().model_dump()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()