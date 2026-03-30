import gradio as gr
from fastapi import FastAPI
import uvicorn
import json
import os
from groq import Groq
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

env = MedicalTriageEnvironment(task_id="triage_basic")

@app.get("/health")
def health():
    return {"status": "Green", "api_connected": client is not None}

@app.post("/reset")
def reset(task_id: str = "triage_basic"):
    env.__init__(task_id=task_id)
    obs = env.reset()
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs

@app.post("/step")
def step(action: MedicalTriageAction):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump() if hasattr(observation, 'model_dump') else observation,
        "reward": float(reward),
        "done": bool(done),
        "info": info if isinstance(info, dict) else {}
    }

@app.get("/state")
def get_state():
    state_data = env.state()
    return state_data.model_dump() if hasattr(state_data, 'model_dump') else state_data

def get_llama_decision(description, bp, hr):
    if not client:
        return {"level": 3, "reasoning": "Missing GROQ_API_KEY."}
                
    prompt = f"Patient: {description}. Vitals: BP {bp}, HR {hr}. Return ONLY JSON: {{'level': <int>, 'reasoning': '<str>'}}"
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return json.loads(chat.choices[0].message.content)
    except:
        return {"level": 3, "reasoning": "Inference error."}

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
    
    while obs is not None:
        ai_output = get_llama_decision(obs.patient_description, obs.vitals_bp, obs.vitals_hr)
        level = ai_output.get("level", 3)
        
        if not isinstance(level, int) or not (1 <= level <= 5):
            level = 3
            
        action = MedicalTriageAction(
            priority_level=level, 
            reasoning=ai_output.get("reasoning", "")
        )
        
        obs, reward, done, info = env.step(action)
        total_score += reward
        icon = "✅" if reward > 0 else "❌"
        
        log_entry = (
            f"{icon} PATIENT: {action.reasoning[:60]}...\n"
            f"   AI CHOICE: Level {level} | REWARD: {reward}\n"
            + "-"*50
        )
        results.append(log_entry)
        
        if done:
            break

    verdict = "🟢 EXCELLENT" if total_score >= 10 else "🟡 SAFE" if total_score >= 0 else "🔴 SAFETY ALERT"
    return "\n".join(results), f"TOTAL SCORE: {total_score} | {verdict}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical Triage AI Evaluation Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            dataset_dropdown = gr.Dropdown(
                choices=["Basic Triage", "Emergency Cases", "Vitals Focus"], 
                value="Basic Triage", 
                label="Select Test Scenario"
            )
            run_btn = gr.Button("🚀 Start Live AI Triage", variant="primary")
            score_display = gr.Label(label="System Verdict")
                    
        with gr.Column(scale=2):
            output_log = gr.Textbox(label="Real-Time Evaluation Logs", lines=20, interactive=False)

    run_btn.click(run_triage_simulation, inputs=[dataset_dropdown], outputs=[output_log, score_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)