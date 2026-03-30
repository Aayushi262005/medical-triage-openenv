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

# 🏥 Clinical Medical Triage Environment (OpenEnv)

> An AI-powered simulation environment for evaluating clinical triage decision-making in emergency department scenarios.

---

## 🌟 Motivation

In emergency departments, the first few minutes of patient assessment are critical. This environment simulates a **clinical triage task**, where an AI agent acts as a triage nurse responsible for prioritizing patients on a scale from **1 (Critical)** to **5 (Non-urgent)**.

Decisions are based on symptom narratives and vital signs (Heart Rate, Blood Pressure). The goal is to evaluate the **clinical reasoning, decision-making, and safety alignment** of AI agents in high-stakes scenarios.

---

## 📁 Project Structure
```plaintext
medical_triage/
├── data/
│   ├── triage_basic.json
│   ├── triage_vitals.json
│   └── triage_emergency.json
├── server/
│   ├── app.py
│   └── medical_triage_environment.py
    └── graders.py  
├── models.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── inference.py
└── README.md
```

---

## 🛠️ Environment Specification

### 📝 Observation Space

At each step, the agent receives a `MedicalTriageObservation` object:

| Field | Type | Description |
|:---|:---|:---|
| `patient_description` | `str` | Narrative describing symptoms, history, and presentation |
| `vitals_hr` | `int` | Heart rate in beats per minute (BPM) |
| `vitals_bp` | `str` | Blood pressure reading (e.g., `"160/100"`) |

### ⚡ Action Space

The agent must return a `MedicalTriageAction` JSON object:

| Field | Type | Description |
|:---|:---|:---|
| `priority_level` | `int` | Priority from `1` (critical) to `5` (non-urgent) |
| `reasoning` | `str` | Clinical justification for the assigned priority |

---

## 🏆 Reward Design & Safety

The reward function prioritizes patient safety and clinical accuracy:

| Outcome | Reward | Condition |
|:---|:---:|:---|
| ✅ Correct Priority | `+1.0` | Exact match |
| 🟡 Close Match | `+0.5` | Within ±1 level |
| 🔴 Critical Safety Penalty | `-2.0` | Level 1 patient assigned Level 3 or higher |
| 🟠 Resource Misallocation | `-0.5` | Non-urgent case incorrectly assigned Level 1 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, recommended)
- An OpenAI or Anthropic API key

---

### Option 1: Docker (Recommended)

**1. Build the image:**
```bash
docker build -t medical-triage .
```

**2. Run the server:**
```bash
docker run -p 7860:7860 medical-triage
```

The environment will be live at `http://localhost:7860`.

---

### Option 2: Local Python Setup

**1. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start the server:**
```bash
python -m server.app
```

> ⚠️ Ensure no other service is using port `7860`.

---

## 🤖 Running the Inference Agent

Once the server is running, evaluate an AI agent against the environment.

### 1. Set Environment Variables

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"   # If using Claude
export MODEL_NAME="gpt-4o"                           # or "claude-3-5-sonnet-20240620"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your_openai_key_here"
$env:MODEL_NAME = "gpt-4o"
```

### 2. Run Inference
```bash
python inference.py
```

---
## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Port 7860 already in use** | Run `lsof -ti:7860 | xargs kill -9` (Mac/Linux) or stop the process in Task Manager (Windows). |
| **Connection Refused** | Ensure the server is fully started before running `inference.py`. It should say `Uvicorn running on http://0.0.0.0:7860`. |
| **API Key Error** | Double-check that `export` or `$env:` was run in the *same terminal session* where you are running the script. |

---
## 📄 License

This project is intended for research and evaluation purposes.