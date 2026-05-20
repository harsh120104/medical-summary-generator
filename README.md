[README.md](https://github.com/user-attachments/files/28067577/README.md)
#  Medical Discharge Summary Generator using LLaMA 3.1

An AI-powered pipeline that automatically extracts structured clinical data from raw medical transcripts and generates professional hospital discharge summaries using Meta's **LLaMA 3.1 8B Instruct** model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Output](#output)
- [Model Details](#model-details)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This project uses a large language model (LLaMA 3.1 8B Instruct) to process raw, unstructured medical transcripts and produce:

1. **Structured JSON** containing key patient data (vitals, diagnosis, medications, etc.)
2. **Validated data objects** using Pydantic schema enforcement
3. **Professional discharge summaries** in natural language
4. **Exported PDF reports** ready for clinical use

The entire pipeline runs on a single GPU using 4-bit quantization (via `bitsandbytes`) to keep memory usage low.

---

## Features

-  **LLM-based extraction** — Uses LLaMA 3.1 8B Instruct to extract structured fields from free-form medical text
-  **Pydantic validation** — Ensures extracted JSON conforms to a defined clinical schema before further processing
-  **Discharge summary generation** — Produces readable, formatted discharge documents from structured data
-  **PDF export** — Saves each summary as a standalone PDF using ReportLab
-  **4-bit quantization** — Loads the model efficiently using NF4 quantization with double quantization enabled

---

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- Hugging Face account with access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Python Dependencies

```
transformers
accelerate
bitsandbytes
sentencepiece
huggingface_hub
pydantic
reportlab
torch
```

---

## Installation

```bash
pip install transformers accelerate bitsandbytes sentencepiece
pip install huggingface_hub pydantic reportlab torch
```

Or if running in a Jupyter/Colab notebook:

```python
!pip install -q transformers accelerate bitsandbytes sentencepiece
!pip install -q huggingface_hub reportlab
```

---

## Setup

### 1. Hugging Face Authentication

You need to authenticate with Hugging Face to access the LLaMA 3.1 model. Run the following in your notebook or script:

```python
from huggingface_hub import login
login()
```

Enter your Hugging Face access token when prompted. Make sure your account has been granted access to the `meta-llama/Llama-3.1-8B-Instruct` model on the Hugging Face Hub.

### 2. GPU Environment

This project is designed to run on a CUDA GPU. It uses `device_map="auto"` to automatically place model layers on available hardware.

---

## Usage

### Running the Full Pipeline

The notebook processes a list of medical transcripts end-to-end:

```python
for i, transcript in enumerate(transcripts, start=1):
    json_output = extract_json(transcript)       # Step 1: Extract structured JSON
    validated   = validate_json(json_output)     # Step 2: Validate schema
    if validated:
        summary = generate_summary(json_output)  # Step 3: Generate discharge summary
        export_pdf(summary, f"summary_{i}.pdf")  # Step 4: Export to PDF
```

### Adding Your Own Transcripts

Add raw medical transcripts to the `transcripts` list in the notebook:

```python
transcripts = [
    """<your raw medical transcript here>""",
    ...
]
```

### Extracted Fields

The model extracts the following fields from each transcript:

| Field | Type | Description |
|---|---|---|
| `patient_name` | str | Full name or descriptive identifier |
| `uhid` | str | Unique Hospital ID |
| `age` | int | Patient age |
| `gender` | str | Patient gender |
| `admission_date` | str | Date of admission |
| `discharge_date` | str | Date of discharge |
| `department` | str | Treating department |
| `primary_diagnosis` | str | Main diagnosis |
| `secondary_diagnosis` | list | Additional diagnoses |
| `complaints` | list | Chief complaints |
| `vitals` | dict | BP, HR, Temp, Height, Weight |
| `investigations` | list | Tests and investigations |
| `treatment` | list | Treatments administered |
| `discharge_medications` | list | Medications on discharge |
| `instructions` | list | Post-discharge instructions |
| `follow_up_date` | str | Scheduled follow-up date |

---

## Project Structure

```
├── Final_LLaMA_3_1.ipynb       # Main notebook
├── summary_1.pdf               # Generated PDF for transcript 1
├── summary_2.pdf               # Generated PDF for transcript 2
├── summary_3.pdf               # Generated PDF for transcript 3
└── README.md                   # Project documentation
```

---

## Output

For each valid transcript, the pipeline produces:

- **Console output** — Raw model output, extracted JSON, and generated summary
- **PDF file** — A formatted discharge summary saved as `summary_<n>.pdf`

Example output fields from a urology transcript:
```json
{
  "patient_name": "61-year-old male patient",
  "department": "Urology",
  "primary_diagnosis": "Elevated Prostate Specific Antigen (PSA)",
  "vitals": { "BP": "120/80", "HR": 72, "Temp": 98.6 },
  "discharge_medications": ["Proscar 5mg once daily"]
}
```

---

## Model Details

| Property | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | `float16` |
| Device mapping | `auto` |
| Max new tokens (extraction) | 700 |
| Max new tokens (summary) | 800 |
| Temperature (extraction) | 0.1 |
| Temperature (summary) | 0.3 |

---

## Limitations

- Accuracy of extraction depends on transcript clarity and structure. Ambiguous or incomplete transcripts may yield `null` values.
- The model requires a CUDA GPU; CPU inference is very slow and not recommended.
- LLaMA 3.1 access on Hugging Face requires manual approval from Meta.
- PDF formatting is basic; further customization with ReportLab may be needed for clinical deployment.
- This tool is intended for research and assistive purposes only and should **not** replace professional clinical documentation.

---

## License

This project is for educational and research use. The LLaMA 3.1 model is subject to [Meta's Llama 3 Community License](https://llama.meta.com/llama3/license/).
