# CoachMeMac

Compact macOS desktop app for simple track coaching workflows.

## Features
- Runner profile + workout input
- Loading + pre-workout summary page
- Live workout page with large timer and split table
- OpenCV motion-based split detection over a virtual line
- Demo mode (fake splits) for easy testing
- Final summary with rule-based coaching feedback
- Optional Llama 3.3 70B API summary (Groq)
- Export text summaries to `data/workout_summaries/`

## macOS Setup
1. Ensure Python 3.10+ is installed.
2. In Terminal:
   ```bash
   cd /workspace/CoachMeMac
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Run
```bash
python main.py
```

## Camera permissions on macOS
- First camera use may trigger a permission prompt.
- If blocked, open **System Settings > Privacy & Security > Camera** and allow Terminal/iTerm/Python app.

## Demo mode
- On input page, enable **Demo mode (no camera)**.
- This generates split events every few seconds.

## Optional LLM (Llama 3.3 70B via API)
In `main.py` set:
- `USE_LLM = True`
- `LLM_PROVIDER = "groq"`
- `LLM_MODEL = "llama-3.3-70b-versatile"`

Set API key (never hardcode):
```bash
export GROQ_API_KEY="your_key_here"
```

If API fails, app automatically falls back to local rule-based summaries.

## Tuning Motion Tracking
In `main.py`, adjust:
- `LINE_POSITION_RATIO` (virtual line position)
- `MOTION_SENSITIVITY` (motion threshold)
- `CROSSING_COOLDOWN_SECONDS` (duplicate trigger cooldown)
- `LINE_ORIENTATION` (`vertical` or `horizontal`)

## Output
Downloaded summaries are saved to:
- `data/workout_summaries/`
