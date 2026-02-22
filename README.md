# 396 Pilot Project

This repo provides a ready-to-run reproduction script that:
1) creates the conda environment,
2) downloads the LoRA adapter from Hugging Face,
3) runs inference + evaluation on **GSM8K test** and **AILuminate**,
4) writes all metrics and per-example predictions to disk.

---

## 0) Prerequisites

- Linux machine with **NVIDIA GPU**
- `conda` installed and available in PATH
- Internet access (downloads the base model + GSM8K dataset + LoRA adapter)

---

## 1) Clone the repo

```bash
git clone https://github.com/AndrewLiu666/396_Pilot_Project.git
cd 396_Pilot_Project
```

---

## 2) Set Hugging Face token

The script downloads the LoRA adapter repo from Hugging Face. Export a token before running.

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

You can create a token at:
https://huggingface.co/settings/tokens

---

## 3) Run everything (single command)
```bash
bash run_all.sh
```

# This command will:
	•	Create conda env pilot from environment.yml (if not already created)
	•	Download LoRA adapter from: AndrewLiu666/qwen2.5-1.5b-lora-pilot-project
# into:
	•	qwen_lora/checkpoint-150
	•	Run GSM8K test evaluation
	•	Run AILuminate evaluation + safeguard classification

---

## 4) Outputs

After the script finishes, the following files will be created:

# GSM8K
    •       Metrics: qwen_lora/test_metrics.json
	•	Predictions: qwen_lora/pred_gsm8k.jsonl
# AILuminate
	•	Metrics: out_ailu_eval/ailuminate_metrics.json
	•	Predictions: out_ailu_eval/pred_ailuminate.jsonl

---

## Notes
	•	GSM8K is downloaded automatically from Hugging Face Datasets the first time you run the script.
	•	The base model is Qwen/Qwen2.5-1.5B-Instruct.
	•	The LoRA adapter is downloaded from Hugging Face and loaded from qwen_lora/checkpoint-150.
	•	If you rerun run_all.sh, it will reuse the existing conda env pilot and will NOT re-download the adapter snapshot into the same folder.