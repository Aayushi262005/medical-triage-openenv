import os
import json
import argparse
import sys
from typing import List

try:
    from server.medical_triage_environment import MedicalTriageEnvironment
    from server.models import MedicalTriageAction
except ImportError:
    pass

from openai import OpenAI

# --- ENV VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

BENCHMARK = "medical_triage"
MAX_STEPS = 50

if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def clamp_reward(reward: float) -> float:
    """Ensure reward is strictly between 0.01 and 0.99"""
    if reward <= 0.0:
        return 0.01
    if reward >= 1.0:
        return 0.99
    return round(reward, 2)


def get_action(obs) -> dict:
    try:
        patient_desc = getattr(obs, 'patient_description', 'No description')
        vitals = f"BP {getattr(obs, 'vitals_bp', 'N/A')}, HR {getattr(obs, 'vitals_hr', 'N/A')}"

        response = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical triage expert. Level 1 is critical/life-threatening, Level 5 is routine. Respond ONLY in valid JSON."
                },
                {
                    "role": "user",
                    "content": f"Triage this patient:\n{patient_desc}\nVitals: {vitals}\n\nReturn JSON format: {{\"priority_level\": int, \"reasoning\": \"string\"}}"
                }
            ],
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        text = str(getattr(obs, "patient_description", "")).lower()
        if any(w in text for w in ["chest", "breath", "unconscious", "stroke", "bleeding"]):
            return {"priority_level": 1, "reasoning": "emergency detected"}
        return {"priority_level": 3, "reasoning": "stable/standard"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", dest="task_id", type=str, default=os.getenv("TASK_ID", "triage_basic"))
    parser.add_argument("--task_id", dest="task_id_alt", type=str, default=None)

    args, _ = parser.parse_known_args()
    task_name = args.task_id_alt if args.task_id_alt else args.task_id

    try:
        env = MedicalTriageEnvironment(task_id=task_name)
    except Exception as e:
        print(f"Failed to load environment: {str(e)}")
        sys.exit(1)

    obs = env.reset()
    rewards: List[float] = []
    success = False
    steps_count = 0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs is None:
                break

            model_output = get_action(obs)

            try:
                level = int(model_output.get("priority_level", 3))
                reason = str(model_output.get("reasoning", "triage step"))

                command = f"priority({level})"
                action = MedicalTriageAction(priority_level=level, reasoning=reason)

                result = env.step(action)

                reward = float(getattr(result, "reward", 0.0))
                done = bool(getattr(result, "done", False))
                err = getattr(result, "last_action_error", None)

                reward = clamp_reward(reward)

                rewards.append(reward)
                steps_count = step

                done_str = "true" if done else "false"
                err_str = "null" if err is None else str(err)

                print(f"[STEP] step={step} action={command} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

                if done:
                    if reward >= 0.99:
                        success = True
                    break

                obs = getattr(result, "observation", None)

            except Exception as e:
                err_str = str(e)
                reward = 0.01
                rewards.append(reward)
                steps_count = step

                print(f"[STEP] step={step} action=error reward={reward:.2f} done=true error={err_str}", flush=True)
                break

    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

        success_str = "true" if success else "false"

        rewards_joined = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"

        print(f"[END] success={success_str} steps={steps_count} rewards={rewards_joined}", flush=True)


if __name__ == "__main__":
    main()