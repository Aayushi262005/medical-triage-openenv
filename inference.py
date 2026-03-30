import os
import json
from openai import OpenAI
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = HF_TOKEN if HF_TOKEN else OPENAI_API_KEY
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def run_task(task_id):
    env = MedicalTriageEnvironment(task_id=task_id)
    obs = env.reset()
    done = False
    total_reward = 0.0
    final_info = {}
    print(f"\n>>> Starting Task: {task_id}")
    max_steps = 10
    step_count = 0

    while not done and obs is not None and step_count < max_steps:
        step_count += 1
        messages = [
            {"role": "system", "content": (
                "You are a professional ER Triage Nurse. Analyze patient data and respond "
                "ONLY with a JSON object containing 'priority_level' (integer 1-5) and 'reasoning' (string)."
            )},
            {"role": "user", "content": (
                f"Patient: {obs.patient_description}. Vitals: HR {obs.vitals_hr}, BP {obs.vitals_bp}."
            )}
        ]
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            action_data = json.loads(content)
            action = MedicalTriageAction(**action_data)
            obs, reward, done, final_info = env.step(action)
            total_reward += reward
            print(f"  Step {step_count}: Assigned Priority {action.priority_level} | Reward: {reward}")
        except Exception as e:
            print(f"  Error during inference for {task_id}: {e}")
            break

    grading_score = final_info.get("grading_score", max(0.0, min(1.0, total_reward / max(step_count, 1))))
    return grading_score


if __name__ == "__main__":
    tasks = ["triage_basic", "triage_vitals", "triage_emergency"]
    results = {}
    print("=== Starting OpenEnv Baseline Inference ===")
    for task in tasks:
        score = run_task(task)
        results[task] = score
        print(f"--- Finished {task} | Final Score: {score} ---")

    print("\n" + "="*40)
    print("BASELINE REPRODUCIBILITY SUMMARY:")
    print(json.dumps(results, indent=4))
    print("="*40)
    print("Inference baseline complete.")