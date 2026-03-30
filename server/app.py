import gradio as gr
from fastapi import FastAPI
import uvicorn
import json
import os
from groq import Groq
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

fastapi_app = FastAPI()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
env = MedicalTriageEnvironment(task_id="triage_basic")

def get_llama_decision(description, bp, hr):
    if not client:
        return {"level": 3, "reasoning": "Missing API Key."}
    prompt = f"Patient: {description}. Vitals: BP {bp}, HR {hr}. Return ONLY JSON: {{'level': <int>, 'reasoning': '<str>'}}"
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return json.loads(chat.choices[0].message.content)
    except:
        return {"level": 3, "reasoning": "Error"}

def run_triage_simulation(dataset_name):
    file_map = {"Basic Triage": "triage_basic", "Emergency Cases": "triage_emergency", "Vitals Focus": "triage_vitals"}
    task_id = file_map.get(dataset_name, "triage_basic")
    env.__init__(task_id=task_id)
    obs = env.reset()
    results = []
    score = 0.0
    while obs is not None:
        ai_output = get_llama_decision(obs.patient_description, obs.vitals_bp, obs.vitals_hr)
        level = ai_output.get("level", 3)
        action = MedicalTriageAction(priority_level=level, reasoning=ai_output.get("reasoning", ""))
        obs, reward, done, info = env.step(action)
        actual = info.get("correct")
        status = info.get("status")
        score = info.get("grading_score")
        icon = "✅" if reward > 0 else "❌"
        if "CRITICAL" in status: icon = "🚨"
        log_entry = f"{icon} AI: Lvl {level} | ACTUAL: Lvl {actual} | STATUS: {status}\nREASON: {action.reasoning[:80]}\n" + "-"*40
        results.append(log_entry)
        if done: break
    verdict = "🟢 EXCELLENT" if score >= 0.8 else "🟡 SAFE" if score >= 0.5 else "🔴 DANGER"
    return "\n".join(results), f"SCORE: {round(score, 2)} | {verdict}"

with gr.Blocks() as demo:
    gr.Markdown("# Medical Triage AI Eval")
    dataset_dropdown = gr.Dropdown(choices=["Basic Triage", "Emergency Cases", "Vitals Focus"], label="Task")
    run_btn = gr.Button("Run Simulation")
    output_log = gr.Textbox(label="Logs", lines=15)
    score_display = gr.Label(label="Result")
    run_btn.click(run_triage_simulation, inputs=[dataset_dropdown], outputs=[output_log, score_display])

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)