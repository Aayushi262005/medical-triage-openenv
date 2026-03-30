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

# --- LLAMA INFERENCE ---
def get_llama_decision(description, bp, hr):
    if not client:
        return {"level": 3, "reasoning": "Missing API Key."}
            
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
        # Strict enforcement of 1-5 scale
        lvl = int(res.get("level", 3))
        res["level"] = max(1, min(5, lvl))
        return res
    except:
        return {"level": 3, "reasoning": "Inference Error"}

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
    total_reward = 0.0
    patient_count = 0
    
    while obs is not None:
        patient_count += 1
        ai_output = get_llama_decision(obs.patient_description, obs.vitals_bp, obs.vitals_hr)
        level = ai_output.get("level", 3)
            
        action = MedicalTriageAction(priority_level=level, reasoning=ai_output.get("reasoning", ""))
        
        # Capture environment data
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        
        actual_lvl = info.get('correct', '?')
        status = info.get('status', 'Standard')
        
        # UI Formatting
        icon = "✅" if reward > 0 else "❌"
        if "CRITICAL" in status: icon = "🚨"
        
        # Truncate description to keep UI clean
        short_desc = (obs.patient_description[:70] + '..') if len(obs.patient_description) > 70 else obs.patient_description

        log_entry = (
            f"{icon} PATIENT: {short_desc}\n"
            f"   AI: Level {level} | CORRECT: Level {actual_lvl}\n"
            f"   REWARD: {reward} | {status}\n"
            + "-"*55
        )
        results.append(log_entry)
        
        if done:
            break
        obs = next_obs

    # --- NORMALIZATION LOGIC ---
    # Convert raw total (e.g. 2.5) to normalized score (e.g. 0.83)
    grading_score = max(0.0, total_reward / patient_count) if patient_count > 0 else 0.0
    final_display = round(grading_score, 2)

    if final_display >= 0.8:
        verdict = "🟢 EXCELLENT"
    elif final_display >= 0.5:
        verdict = "🟡 SAFE"
    else:
        verdict = "🔴 SAFETY ALERT"
        
    return "\n".join(results), f"GRADING SCORE: {final_display} | {verdict}"

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