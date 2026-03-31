---
title: Medical Triage Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - medical-ai
---
 
# Clinical Medical Triage Environment (OpenEnv)
 
An AI-powered simulation environment for evaluating clinical triage decision-making in emergency department scenarios.
 
---
 
## Motivation
 
In emergency departments, the first few minutes of patient assessment are critical. This environment simulates a real-world clinical triage task, where an AI agent acts as a triage nurse responsible for prioritizing patients on a scale from **1 (Critical)** to **5 (Non-urgent)**.
 
Decisions are based on symptom narratives and vital signs (Heart Rate, Blood Pressure). The goal is to evaluate the **clinical reasoning, decision-making, and safety alignment** of AI agents in high-stakes scenarios.
 
---
 
## Tasks
 
The environment includes three graded tasks with increasing difficulty:
 
- 🟢 **triage_basic (Easy)**: Clear and straightforward cases  
- 🟡 **triage_vitals (Medium)**: Requires interpretation of vital signs  
- 🔴 **triage_emergency (Hard)**: Subtle, high-risk cases  
 
Each task includes a deterministic grader producing a score between **0.0 and 1.0**.
 
---
 
## Project Structure
 
```plaintext
medical_triage/
├── data/
│   ├── triage_basic.json
│   ├── triage_vitals.json
│   └── triage_emergency.json
├── server/
│   ├── app.py
│   ├── medical_triage_environment.py
│   └── graders.py
├── models.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── inference.py
└── README.md
```
 
---
 
## 🛠️ Environment Specification
 
### Observation Space
 
At each step, the agent receives a `MedicalTriageObservation` object:
 
| Field | Type | Description |
|:---|:---|:---|
| `patient_description` | `str` | Narrative describing symptoms, history, and presentation |
| `vitals_hr` | `int` | Heart rate in beats per minute (BPM) |
| `vitals_bp` | `str` | Blood pressure reading (e.g., `"160/100"`) |
| `current_waiting_room_count` | `int` | Number of remaining patients |
 
### Action Space
 
The agent must return a `MedicalTriageAction` JSON object:
 
| Field | Type | Description |
|:---|:---|:---|
| `priority_level` | `int` | Priority from `1` (critical) to `5` (non-urgent) |
| `reasoning` | `str` | Clinical justification for the assigned priority |
 
---
 
## OpenEnv Interface
 
- `reset()` → Returns initial observation  
- `step(action)` → Returns `(observation, reward, done, info)`  
- `state()` → Returns current state  
 
---
 
## Reward Design & Safety
 
| Outcome | Reward | Condition |
|:---|:---:|:---|
| ✅ Correct Priority | `+1.0` | Exact match |
| 🟡 Close Match | `+0.5` | Within ±1 level |
| 🔴 Critical Safety Penalty | `-2.0` | Level 1 patient assigned Level ≥3 |
| 🟠 Resource Misallocation | `-0.5` | Non-urgent case assigned Level 1 |
 
---
 
## Getting Started
 
### Prerequisites
 
- Python 3.9+
- Docker (recommended)
- API access (OpenAI-compatible)
 
---
 
## Docker Setup
 
```bash
docker build -t medical-triage .
docker run -p 7860:7860 medical-triage
```
 
The environment will be available at:
 
```bash
http://localhost:7860
```
 
---
 
## Local Setup
 
```bash
python -m venv venv
 
# macOS/Linux
source venv/bin/activate
 
# Windows (PowerShell)
venv\Scripts\Activate.ps1
 
pip install -r requirements.txt
python -m server.app
```
 
---
 
## Running the Inference Agent
 
Once the server is running, evaluate an AI agent against the environment.
 
### 1. Set Environment Variables
 
**Windows (PowerShell):**
 
```bash
$env:HF_TOKEN="your_hf_token"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
```
 
**Using OpenAI:**
 
```bash
export OPENAI_API_KEY="your_openai_key"
export MODEL_NAME="gpt-4o-mini"
```
 
### 2. Run Inference
 
```bash
python inference.py
```
 
---
 
## Example Output
 
```
Running task: triage_basic
Score: 1.00
 
Running task: triage_vitals
Score: 0.83
 
Running task: triage_emergency
Score: 1.00
```
 
---
 
## Troubleshooting
 
| Issue | Solution |
|:---|:---|
| **Port 7860 already in use** | Run `lsof -ti:7860 \| xargs kill -9` (Mac/Linux) or stop the process in Task Manager (Windows). |
| **Connection Refused** | Ensure the server is fully started before running `inference.py`. It should say `Uvicorn running on http://0.0.0.0:7860`. |
| **API Key Error** | Double-check that `export` or `$env:` was run in the *same terminal session* where you are running the script. |
 
---
 
## License
 
This project is intended for research, evaluation, and educational purposes.
 