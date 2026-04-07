import os
import json
from openai import OpenAI

from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "dummy-key"
    )
except Exception as e:
    print(f"[ERROR] Client init failed: {e}", flush=True)
    client = None

TASK_NAME = "medical_triage"
MAX_STEPS = 50


def get_model_action(prompt):
    try:
        # If no API key → skip model call
        if not API_KEY or client is None:
            return {"priority_level": 3, "reasoning": "No API fallback"}

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content
        return json.loads(text)

    except Exception as e:
        print(f"[WARN] Model failed: {e}", flush=True)
        return {"priority_level": 3, "reasoning": "Fallback"}


def run_task(task_id):
    try:
        env = MedicalTriageEnvironment(task_id=task_id)
        obs = env.reset()
    except Exception as e:
        print(f"[ERROR] Env failed: {e}", flush=True)
        return 0.0

    total_reward = 0.0
    steps = 0

    print(f"[START] task={task_id}", flush=True)

    while obs is not None and steps < MAX_STEPS:
        steps += 1

        try:
            prompt = f"""
Patient: {obs.patient_description}
Vitals: BP {obs.vitals_bp}, HR {obs.vitals_hr}

Return ONLY JSON:
{{
    "priority_level": integer (1-5),
    "reasoning": "short explanation"
}}
"""
        except Exception:
            prompt = "Return JSON with priority_level 1-5"

        model_output = get_model_action(prompt)

        level = model_output.get("priority_level", 3)
        if not isinstance(level, int) or level < 1 or level > 5:
            level = 3

        action = MedicalTriageAction(
            priority_level=level,
            reasoning=model_output.get("reasoning", "")
        )

        try:
            result = env.step(action)
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", flush=True)
            break

        total_reward += result.reward

        print(f"[STEP] step={steps} reward={float(result.reward)}", flush=True)

        if result.done:
            break

        obs = result.observation

    score = max(0.01, min(0.99, (total_reward + 2) / 3))

    print(f"[END] task={task_id} score={float(score)} steps={steps}", flush=True)

    return score


def main():
    try:
        run_task("triage_basic")
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)


if __name__ == "__main__":
    main()