import gradio as gr
from fastapi import FastAPI
import uvicorn
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()
env = MedicalTriageEnvironment(task_id="triage_basic")

def run_triage_simulation(dataset_name):
    file_map = {
        "Basic Triage": "triage_basic",
        "Emergency Cases": "triage_emergency",
        "Vitals Focus": "triage_vitals"
    }
    
    task_id = file_map.get(dataset_name, "triage_basic")
    
    # Re-initialize the environment with the selected dataset
    env.__init__(task_id=task_id)
    
    obs = env.reset()
    results = []
    total_score = 0.0
    done = False
    
    while not done:
        current_desc = obs.patient_description
        current_bp = obs.vitals_bp
        current_hr = obs.vitals_hr
        
        # Simulated Agent Action
        action = MedicalTriageAction(
            priority_level=3, 
            reasoning=f"Automated assessment of {current_desc}."
        )
        
        step_result = env.step(action)
        
        reward = step_result.reward
        total_score += reward
        status = step_result.info.get("status", "Normal")
        
        icon = "❌" if "VIOLATION" in status else "✅"
        log_entry = (
            f"{icon} PATIENT: {current_desc}\n"
            f"   📊 VITALS: BP {current_bp} | HR {current_hr}\n"
            f"   🎯 ACTION: Level {action.priority_level} | REWARD: {reward}\n"
            f"   📝 FEEDBACK: {status}\n"
            + "-"*40
        )
        results.append(log_entry)
        
        if step_result.done:
            done = True
        else:
            obs = step_result.observation

    if total_score >= 2.0:
        verdict = "🟢 ELITE PERFORMANCE"
    elif total_score >= 0:
        verdict = "🟡 PASS"
    else:
        verdict = "🔴 SAFETY FAILURE"

    return "\n".join(results), f"TOTAL SCORE: {total_score} | {verdict}"

# --- GRADIO DASHBOARD ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical Triage AI Evaluation Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            dataset_dropdown = gr.Dropdown(
                choices=["Basic Triage", "Emergency Cases", "Vitals Focus"],
                value="Basic Triage",
                label="Select Test Scenario"
            )
            run_btn = gr.Button("🚀 Run AI Agent Trial", variant="primary")
            score_display = gr.Label(label="Final Evaluation Score")
            
        with gr.Column(scale=2):
            output_log = gr.Textbox(label="Live Environment Logs", lines=15)

    run_btn.click(run_triage_simulation, inputs=[dataset_dropdown], outputs=[output_log, score_display])

# --- API ENDPOINTS ---
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