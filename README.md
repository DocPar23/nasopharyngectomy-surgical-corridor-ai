# nasopharyngectomy-surgical-corridor-ai
nvidia vista 
# AI-Powered Surgical Corridor Finder for Nasopharyngectomy

## What This Does
This tool reads CT scans of patients with nasopharyngeal carcinoma and 
automatically identifies surgical safe zones — helping surgeons know 
exactly where to operate and what to avoid.

## How It Works
- Input: Patient CT scan
- Processing: AI model built using MONAI (NVIDIA's medical imaging framework) 
  and Python, developed with Claude Code
- Output: Colour-coded surgical map overlaid on the CT scan
  - 🟢 Green — Safe to operate
  - 🟡 Yellow — Proceed with caution
  - 🔴 Red — Avoid (critical structures)

## Why It Matters
Nasopharyngectomy is one of the most technically demanding surgeries in 
head and neck oncology. One wrong move near critical vessels or nerves 
can be catastrophic. This tool gives surgeons a real-time AI-powered 
safety layer before and during surgical planning.

## Built With
- Python
- MONAI by NVIDIA
- 3D Slicer
- Claude Code

## Research
This work was selected as a poster presentation at FHNO 2025 
(Federation of Head and Neck Oncologists) and is currently under publication.

## Developer
Dr. Parnini — Head and neck surgeon, Head & Neck Surgical Oncology researcher, and AI builder.
