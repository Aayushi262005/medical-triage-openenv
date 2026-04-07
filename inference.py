import os
import json
import argparse
import sys
from openai import OpenAI

try:
    from server.medical_triage_environment import MedicalTriageEnvironment
    from models import MedicalTriageAction
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_action(obs):
    try:
        patient_desc = getattr(obs, 'patient_description', 'No description')
        vitals = f"BP {getattr(obs, 'vitals_bp', 'N/A')}, HR {getattr(obs, 'vitals_hr', 'N/A')}"
        prompt = f"Patient: {patient_desc}\nVitals: {vitals}\nReturn JSON: {{\"priority_level\": 1-5, \"reasoning\": \"short\"}}"
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        text = str(getattr(obs, "patient_description", "")).lower()
        if "chest pain" in text or "unconscious" in text:
            return {"priority_level": 1, "reasoning": "emergency"}
        return {"priority_level": 3, "reasoning": "stable"}

def main():
    if hasattr(main, "_has_run"):
        return
    main._has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", dest="task_id", type=str, default=os.getenv("TASK_ID", "triage_basic"))
    args, _ = parser.parse_known_args()
    
    task_name = args.task_id
    env_name = "medical_triage"
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}", flush=True)

    try:
        env = MedicalTriageEnvironment(task_id=task_name)
        obs = env.reset()

        while obs is not None and steps < 50:
            steps += 1
            model_output = get_action(obs)
            level = int(model_output.get("priority_level", 3))
            reason = str(model_output.get("reasoning", "triage"))

            try:
                action = MedicalTriageAction(priority_level=level, reasoning=reason)
                result = env.step(action)
                
                reward = float(getattr(result, "reward", 0.0))
                done = bool(getattr(result, "done", False))
                err = getattr(result, "last_action_error", None)
                error_msg = "null" if err is None else str(err)
                
                rewards.append(f"{reward:.2f}")
                print(f"[STEP] step={steps} action=priority({level}) reward={reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)
                
                if done:
                    if reward >= 1.0: success = True
                    break
                obs = getattr(result, "observation", None)
            except Exception as e:
                break

        if hasattr(env, "close"):
            env.close()

    except Exception:
        pass

    if not rewards: rewards = ["0.00"]
    print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(rewards)}", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()