import gradio as gr
from fastapi import FastAPI
import uvicorn
import json
import os
from groq import Groq
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

app = FastAPI()

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
env = MedicalTriageEnvironment(task_id="triage_basic")

# --- API ENDPOINTS ---
@app.get("/health")
def health():
    return {"status": "Green", "message": "Active", "api_connected": client is not None}

@app.post("/reset")
def reset(task_id: str = "triage_basic"):
    env.__init__(task_id=task_id)
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

# --- LLAMA INFERENCE ---
def get_llama_decision(description, bp, hr):
    if not client:
        return {"level": 3, "reasoning": "Missing GROQ_API_KEY in Secrets."}
            
    prompt = f"Patient: {description}. Vitals: BP {bp}, HR {hr}. Return ONLY JSON: {{'level': <int>, 'reasoning': '<str>'}}"
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return json.loads(chat.choices[0].message.content)
    except Exception as e:
        return {"level": 3, "reasoning": f"AI Error: {str(e)}"}

# --- UI LOGIC ---
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
    
    # Loop through patients
    while obs is not None:
        ai_output = get_llama_decision(obs.patient_description, obs.vitals_bp, obs.vitals_hr)
        level = ai_output.get("level", 3)
        
        # Guardrail: Ensure level is 1-5
        if not isinstance(level, int) or level < 1 or level > 5:
            level = 3
            
        action = MedicalTriageAction(priority_level=level, reasoning=ai_output.get("reasoning", ""))
        step_result = env.step(action)
        
        total_score += step_result.reward
        icon = "❌" if step_result.reward < 0 else "✅"
        
        log_entry = (
            f"{icon} PATIENT: {obs.patient_description}\n"
            f"   AI CHOICE: Level {level} | CORRECT: Level {step_result.info.get('correct')}\n"
            f"   REWARD: {step_result.reward} | 📝 FEEDBACK: {step_result.info.get('status')}\n"
            + "-"*50
        )
        results.append(log_entry)
        
        if step_result.done:
            break
        obs = step_result.observation

    # Final Verdict Calculation
    if total_score >= 10:
        verdict = "🟢 EXCELLENT"
    elif total_score >= 0:
        verdict = "🟡 SAFE"
    else:
        verdict = "🔴 SAFETY ALERT"
        
    return "\n".join(results), f"TOTAL SCORE: {total_score} | {verdict}"

# --- GRADIO INTERFACE ---
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

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Standard HF port configuration
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)