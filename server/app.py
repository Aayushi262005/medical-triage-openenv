import gradio as gr
from fastapi import FastAPI
import uvicorn
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()
env = MedicalTriageEnvironment(task_id="triage_basic")

@app.get("/health")
def health():
    return {"status": "Green", "message": "Active"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump() if obs else {"error": "Reset failed"}

@app.post("/step")
def step(action: MedicalTriageAction):
    result = env.step(action)
    return {
        "observation": result.observation.model_dump() if result.observation else None,
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

def run_triage_simulation(dataset_name):
    file_map = {
        "Basic Triage": "triage_basic",
        "Emergency Cases": "triage_emergency",
        "Vitals Focus": "triage_vitals"
    }
    task_id = file_map.get(dataset_name, "triage_basic")
    env.__init__(task_id=task_id)
    
    obs = env.reset()
    results = []
    total_score = 0.0
    done = False
    
    while not done and obs is not None:
        current_desc = obs.patient_description
        action = MedicalTriageAction(priority_level=3, reasoning="Dashboard Test")
        step_result = env.step(action)
        
        reward = step_result.reward
        total_score += reward
        correct_level = step_result.info.get("correct", "N/A")
        
        icon = "❌" if reward < 0 else "✅"
        results.append(f"{icon} PATIENT: {current_desc[:50]}...\n   🤖 ACTION: Level 3 | ✅ CORRECT: Level {correct_level} | 💰 REWARD: {reward}\n" + "-"*30)
        
        if step_result.done: done = True
        else: obs = step_result.observation

    return "\n".join(results), f"SCORE: {total_score}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical Triage AI Evaluation Dashboard")
    with gr.Row():
        with gr.Column(scale=1):
            dataset_dropdown = gr.Dropdown(choices=["Basic Triage", "Emergency Cases", "Vitals Focus"], value="Basic Triage", label="Dataset")
            run_btn = gr.Button("🚀 Run Dashboard Trial", variant="primary")
            score_display = gr.Label(label="Result")
        with gr.Column(scale=2):
            output_log = gr.Textbox(label="Logs", lines=15)
    run_btn.click(run_triage_simulation, inputs=[dataset_dropdown], outputs=[output_log, score_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)