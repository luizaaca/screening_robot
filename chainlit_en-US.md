# Screening Robot Agent

Stateful clinical screening assistant powered by LangGraph, SQLite patient lookup, and configurable clinical-model backends.

## What you can do

- Ask how to use the assistant.
- Find a patient by fictional security number or by name.
- Select one patient from an enumerated disambiguation list.
- Clear the active patient context.
- Describe symptoms and request likely conditions or relevant exams.
- Upload or reference a video for expression, posture, and transcription analysis.

## Good prompts to try

- `Find patient Maria Silva`
- `Lookup patient 12003456`
- `Patient 55667788 has fatigue and frequent urination`
- `Analyze this video with video_path=concepts_video/sample.mp4`
- `Clear active patient`

## Demo note

If the patient database is empty, run `python seed_demo_data.py` in the project root before testing lookup flows.

## Disclaimer

This application is for software prototyping and workflow validation only. It does not replace professional medical evaluation, diagnosis, or emergency care.
