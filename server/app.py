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
    return obs.model_dump() if obs else {"error": "Reset failed"}

@app.post("/step")
def step(action: MedicalTriageAction):
    # Ensure action matches Pydantic model expectations
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump() if obs else None,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

def get_llama_decision(description, bp, hr):
    if not client:
        return {"level": 3, "reasoning": "Missing API Key."}
            
    # Explicitly instruct 1-5 scale to prevent "Level 7" errors
    prompt = (
        f"Patient: {description}. Vitals: BP {bp}, HR {hr}. "
        f"Assign Triage Level: 1 (Critical) to 5 (Non-urgent). "
        f"Return ONLY JSON: {{'level': <int>, 'reasoning': '<str>'}}"
    )
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        res = json.loads(chat.choices[0].message.content)
        # Force integer and 1-5 range clamp
        lvl = int(res.get("level", 3))
        res["level"] = max(1, min(5, lvl))
        return res
    except:
        return {"level": 3, "reasoning": "AI Inference Error."}

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
    
    while obs is not None:
        ai_output = get_llama_decision(obs.patient_description, obs.vitals_bp, obs.vitals_hr)
        level = ai_output["level"]
            
        action = MedicalTriageAction(priority_level=level, reasoning=ai_output.get("reasoning", ""))
        
        # Step the environment
        next_obs, reward, done, info = env.step(action)
        
        correct_lvl = info.get('correct', '?')
        status = info.get('status', 'Standard')
        score = info.get('grading_score', 0.0) # Normalized 0.0-1.0
        
        # UI Formatting
        icon = "✅" if reward > 0 else "❌"
        if "CRITICAL" in status: icon = "🚨"
        
        # Truncate description to prevent massive scrolling
        short_desc = (obs.patient_description[:75] + '..') if len(obs.patient_description) > 75 else obs.patient_description

        log_entry = (
            f"{icon} PATIENT: {short_desc}\n"
            f"   AI: Level {level} | CORRECT: Level {correct_lvl}\n"
            f"   REWARD: {reward} | {status}\n"
            + "-"*60
        )
        results.append(log_entry)
        
        if done:
            break
        obs = next_obs

    # Verdict based on normalized grading score (0.0 - 1.0)
    verdict = "🟢 EXCELLENT" if score >= 0.8 else "🟡 SAFE" if score >= 0.5 else "🔴 SAFETY ALERT"
        
    return "\n".join(results), f"GRADING SCORE: {round(score, 2)} | {verdict}"

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
            output_log = gr.Textbox(
                label="Evaluation Logs (AI vs Ground Truth)", 
                lines=18, 
                interactive=False,
                autoscroll=True
            )

    run_btn.click(run_triage_simulation, inputs=[dataset_dropdown], outputs=[output_log, score_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)