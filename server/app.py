import gradio as gr
from fastapi import FastAPI
import uvicorn
import json
import os
from openai import OpenAI

from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction
from server.graders import MedicalTriageGrader

app = FastAPI()

# --- CONFIGURATION ---
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

# --- MODEL INFERENCE ---
def get_model_decision(description, bp, hr):
    prompt = f"""
    Patient: {description}
    Vitals: BP {bp}, HR {hr}
    Assign Triage Level: 1 (Critical) to 5 (Non-urgent).
    Return ONLY JSON:
    {{
        "priority_level": integer (1-5),
        "reasoning": "short explanation"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        res = json.loads(response.choices[0].message.content)

        lvl = int(res.get("priority_level", 3))
        lvl = max(1, min(5, lvl))

        return {
            "level": lvl,
            "reasoning": res.get("reasoning", "")
        }

    except Exception as e:
        print("Inference error:", e)
        return {"level": 3, "reasoning": "Fallback"}

env = MedicalTriageEnvironment(task_id="triage_basic")

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump() if obs else {}

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

# --- UI LOGIC ---
def run_triage_simulation(dataset_name):
    file_map = {
        "Basic Triage": "triage_basic",
        "Emergency Cases": "triage_emergency",
        "Vitals Focus": "triage_vitals"
    }

    task_id = file_map.get(dataset_name, "triage_basic")

    env = MedicalTriageEnvironment(task_id=task_id)

    obs = env.reset()
    results = []
    episode_results = []   
    patient_count = 0

    while obs is not None:
        patient_count += 1

        ai_output = get_model_decision(
            obs.patient_description,
            obs.vitals_bp,
            obs.vitals_hr
        )

        level = ai_output.get("level", 3)

        action = MedicalTriageAction(
            priority_level=level,
            reasoning=ai_output.get("reasoning", "")
        )

        result = env.step(action)
        episode_results.append(result)   # ✅ TRACK

        next_obs = result.observation
        reward = result.reward
        done = result.done
        info = result.info

        actual_lvl = info.get('correct_level', '?')
        status = info.get('status', 'Standard')

        icon = "✅" if reward > 0 else "❌"
        if info.get("is_critical"):
            icon = "🚨"

        short_desc = (obs.patient_description[:70] + '..') if len(obs.patient_description) > 70 else obs.patient_description

        log_entry = (
            f"{icon} PATIENT: {short_desc}\n"
            f"   AI: Level {level} | CORRECT: Level {actual_lvl}\n"
            f"   REWARD: {reward} | {status}\n"
            + "-" * 55
        )

        results.append(log_entry)

        if done:
            break

        obs = next_obs

    # ✅ USE NEW GRADER
    grader = MedicalTriageGrader()
    grading_score = grader.grade(episode_results)
    final_display = round(grading_score, 2)

    if final_display >= 0.8:
        verdict = "🟢 EXCELLENT"
    elif final_display >= 0.5:
        verdict = "🟡 SAFE"
    else:
        verdict = "🔴 SAFETY ALERT"

    return "\n".join(results), f"GRADING SCORE: {final_display} | {verdict}"

# --- GRADIO UI ---
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

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()