import os
import json
from openai import OpenAI

from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not API_KEY:
    raise ValueError("API key not found. Set HF_TOKEN or OPENAI_API_KEY.")

if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable not set.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

TASK_NAME = "medical_triage"

MAX_STEPS = 50


def get_model_action(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content
        return json.loads(text)

    except Exception:
        return {"priority_level": 3, "reasoning": "Fallback"}


def run_task(task_id):
    env = MedicalTriageEnvironment(task_id=task_id)
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    print(f"[START] task={task_id}", flush=True)

    while obs is not None and steps < MAX_STEPS:
        steps += 1

        prompt = f"""
Patient: {obs.patient_description}
Vitals: BP {obs.vitals_bp}, HR {obs.vitals_hr}

Return ONLY JSON:
{{
    "priority_level": integer (1-5),
    "reasoning": "short explanation"
}}
"""

        model_output = get_model_action(prompt)

        level = model_output.get("priority_level", 3)
        if not isinstance(level, int) or level < 1 or level > 5:
            level = 3

        action = MedicalTriageAction(
            priority_level=level,
            reasoning=model_output.get("reasoning", "")
        )

        result = env.step(action)
        total_reward += result.reward

        print(f"[STEP] step={steps} reward={float(result.reward)}", flush=True)

        if result.done:
            break

        obs = result.observation

    score = max(0.0, min(1.0, (total_reward + 2) / 3))

    print(f"[END] task={task_id} score={float(score)} steps={steps}", flush=True)

    return score


def main():
    run_task("triage_basic")


if __name__ == "__main__":
    main()