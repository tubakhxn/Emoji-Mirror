# Emoji Game

Emoji Game is a small real-time demo that captures your webcam and converts a single detected hand gesture into an emoji overlay.

Written and coded by: Tuba Khan
GitHub: https://github.com/tubakhxnso

Supported gestures
- Fist → ✊
- Open palm → 🖐
- Peace sign → ✌️
- Thumbs up → 👍
- OK sign → 👌

Requirements
- Python 3.8+
- Packages: `opencv-python`, `mediapipe`. `pillow` is optional but recommended for nicer emoji rendering on Windows.

Quick install (PowerShell)

```powershell
pip install -r requirements.txt
```

Or install only the required packages:

```powershell
pip install opencv-python mediapipe
# optional for colored emoji rendering
pip install pillow
```

Run

```powershell
python "c:\Users\Tuba Khan\Downloads\emoji\emoji_gestures.py"
```

Usage
- The webcam window shows hand landmarks and an emoji overlay when a supported gesture is detected.
- Press `q` to quit.

License
- This project is licensed under the MIT License (see `LICENSE`).

Notes & tips
- Pillow + an emoji-capable system font (e.g. Segoe UI Emoji on Windows) will make the emoji render in color. If Pillow or such a font is not available, the script falls back to OpenCV text rendering.
- Gesture recognition is heuristic-based (tip vs pip checks and distances). If detection is noisy, try improving lighting and keeping the hand roughly facing the camera.

Files
- `emoji_gestures.py` – main script to run.
- `requirements.txt` – Python dependencies.
- `LICENSE` – MIT license for this project.

If you'd like me to add PNG emoji assets for consistent cross-platform rendering, or a small settings UI for thresholds and camera index, I can add that next.