import gradio as gr
from fastapi import FastAPI
import uvicorn
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()
env = MedicalTriageEnvironment(task_id="triage_basic")

def run_triage_simulation():
    obs = env.reset()
    results = []
    total_score = 0.0
    done = False
    
    while not done:
        current_desc = obs.patient_description
        # This calls your class logic (including the -2.0 and -0.5 penalties)
        action = MedicalTriageAction(priority_level=3, reasoning="AI trial assessment.")
        step_result = env.step(action)
        
        reward = step_result.reward
        total_score += reward
        status = step_result.info.get("status", "Standard Triage")
        
        # Color coding the log for the judge
        prefix = "❌" if "VIOLATION" in status else "✅"
        results.append(f"{prefix} Patient: {current_desc}\n   Action: Level {action.priority_level} | Reward: {reward}\n   Feedback: {status}\n" + "-"*25)
        
        if step_result.done:
            done = True
        else:
            obs = step_result.observation

    verdict = "🟢 HIGH SAFETY" if total_score >= 1.5 else "🔴 SAFETY ALERT"
    return "\n".join(results), f"TOTAL SCORE: {total_score} | {verdict}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical Triage AI Dashboard")
    with gr.Row():
        with gr.Column(scale=2):
            output_log = gr.Textbox(label="Live Environment Logs (Internal Penalties Active)", lines=12)
        with gr.Column(scale=1):
            score_display = gr.Label(label="Final Agent Score")
            run_btn = gr.Button("🚀 Run AI Agent Trial", variant="primary")
    run_btn.click(run_triage_simulation, outputs=[output_log, score_display])

@app.get("/health")
def health():
    return {"message": "Medical Triage Environment is Active", "status": "Green"}

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

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)