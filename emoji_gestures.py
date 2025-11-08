"""
Emoji Game — Real-time Hand Gesture → Emoji converter

Emoji Game captures your webcam and converts one detected hand gesture into an emoji overlay in real-time.

Written and coded by: Tuba Khan
GitHub: https://github.com/tubakhxnso
License: MIT (see LICENSE)

Supported gestures:
 - Fist → ✊
 - Open palm → 🖐
 - Peace → ✌️
 - Thumbs up → 👍
 - OK sign → 👌

Requirements:
 - Python 3.8+
 - opencv-python, mediapipe
 - pillow (optional, recommended for colored emoji rendering)

Run until the user presses 'q' to quit the webcam window.

The program contains clear comments explaining each part.
"""

import math
import time
import cv2
import numpy as np
import mediapipe as mp

# Try to import Pillow for nicer emoji rendering.
# If Pillow is not available, we'll fallback to cv2.putText (may not render colored emoji).
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Common emoji-capable font paths to try when rendering with PIL
EMOJI_FONT_PATHS = [
    "C:\\Windows\\Fonts\\seguiemj.ttf",  # Segoe UI Emoji (Windows)
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
    "/usr/share/fonts/truetype/emoji/DejaVuSans.ttf",  # some Linux
]


# ---------- Utilities for landmark conversions and geometry ----------

def lm_to_point(lm, frame_w, frame_h):
    """Convert a MediaPipe normalized landmark to pixel coordinates (x, y)."""
    return int(lm.x * frame_w), int(lm.y * frame_h)


def distance(a, b):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------- Gesture recognition logic ----------

def fingers_status(landmarks, handedness_label, img_w, img_h):
    """
    Determine which fingers are 'up' (extended).
    Returns dict with keys: thumb, index, middle, ring, pinky (True/False).
    landmarks: list of 21 mediapipe landmark objects (normalized).
    handedness_label: 'Left' or 'Right' (as reported by MediaPipe).
    """
    # Convert relevant landmarks to pixel coordinates for simpler comparisons
    pts = {i: lm_to_point(landmarks[i], img_w, img_h) for i in range(len(landmarks))}
    # For index, middle, ring, pinky: compare tip y position to pip y (since y grows downward)
    finger_tips = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
    finger_pips = {'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
    status = {}
    for name in finger_tips:
        tip = pts[finger_tips[name]]
        pip = pts[finger_pips[name]]
        # If tip is above pip (smaller y), we treat it as extended.
        status[name] = tip[1] < pip[1]

    # Thumb detection: improved heuristic that uses palm center and hand scale.
    # Compute a rough hand "diagonal" and palm center to make thresholds scale-invariant.
    xs_all = [p[0] for p in pts.values()]
    ys_all = [p[1] for p in pts.values()]
    min_x_a, max_x_a = min(xs_all), max(xs_all)
    min_y_a, max_y_a = min(ys_all), max(ys_all)
    box_w_a = max_x_a - min_x_a
    box_h_a = max_y_a - min_y_a
    diag_a = math.hypot(box_w_a, box_h_a)
    if diag_a < 1:
        diag_a = max(img_w, img_h) * 0.2

    # Palm center: average of wrist and MCP joints (5,9,13,17) gives a stable center
    palm_pts = [pts[0], pts[5], pts[9], pts[13], pts[17]]
    pc_x = int(sum([p[0] for p in palm_pts]) / len(palm_pts))
    pc_y = int(sum([p[1] for p in palm_pts]) / len(palm_pts))
    palm_center = (pc_x, pc_y)

    wrist = pts[0]
    thumb_tip = pts[4]
    thumb_ip = pts[3]

    # Distance from thumb tip to palm center relative to hand size
    dist_thumb_palm = distance(thumb_tip, palm_center)
    # Use a fraction of the diagonal as threshold (tunable)
    thumb_dist_thresh = 0.32 * diag_a

    # Also consider vertical separation (thumb above IP) and lateral offset
    dx_tip_ip = abs(thumb_tip[0] - thumb_ip[0])
    dy_tip_ip = thumb_ip[1] - thumb_tip[1]

    # Thresholds scaled by diagonal
    lateral_thresh = 0.12 * diag_a
    vertical_thresh = 0.08 * diag_a

    # Thumb extended if it lies far from palm center AND shows lateral/vertical separation
    thumb_extended = (dist_thumb_palm > thumb_dist_thresh) and ((dx_tip_ip > lateral_thresh) or (dy_tip_ip > vertical_thresh))

    # As a safety, require thumb tip not extremely close to wrist
    if distance(thumb_tip, wrist) < 0.04 * max(img_w, img_h):
        thumb_extended = False

    status['thumb'] = thumb_extended
    return status, pts


def detect_gesture(landmarks, handedness_label, img_w, img_h):
    """
    Recognize gestures from landmarks. Returns (emoji_char, name).
    The supported gestures:
      Fist -> 44a
      Open palm -> 
      Peace -> ✌️ (index + middle up)
      Thumbs up -> 44d (only thumb up)
      OK -> 44c (thumb and index tips close)
    If none match, returns (None, 'Unknown').
    """
    status, pts = fingers_status(landmarks, handedness_label, img_w, img_h)
    # Count how many fingers (not counting thumb) are up
    fingers_up_count = sum([status['index'], status['middle'], status['ring'], status['pinky']])

    # Compute bounding box & diagonal to get scale for thresholds
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    box_w = max_x - min_x
    box_h = max_y - min_y
    diag = math.hypot(box_w, box_h) if box_w and box_h else max(img_w, img_h) * 0.2

    # Gesture: Fist -> no fingers extended (and thumb not extended)
    # Fist: all finger tips are close to their corresponding MCP/PIP joints (folded)
    # As an additional check, ensure tips are near the palm center.
    palm_center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
    tip_distances = [distance(pts[i], palm_center) for i in [8, 12, 16, 20]]
    # If all non-thumb fingers are folded (not extended) and thumb is not extended, it's a fist
    if (not status['index'] and not status['middle'] and not status['ring'] and not status['pinky'] and not status['thumb']):
        return '✊', 'Fist'

    # Open palm -> all five fingers extended
    if status['thumb'] and status['index'] and status['middle'] and status['ring'] and status['pinky']:
        return '🖐', 'Open Palm'

    # Peace -> index and middle extended, ring and pinky folded
    if status['index'] and status['middle'] and (not status['ring']) and (not status['pinky']):
        return '✌️', 'Peace'

    # Thumbs up -> thumb extended and other fingers folded and thumb tip is higher than wrist (upwards)
    wrist = pts[0]
    thumb_tip = pts[4]
    # Check thumb is extended and others folded
    if status['thumb'] and (not status['index']) and (not status['middle']) and (not status['ring']) and (not status['pinky']):
        # Check that thumb tip is above wrist (smaller y) by some margin
        if thumb_tip[1] < wrist[1] - 20:
            return '👍', 'Thumbs Up'
        # Also accept lateral thumbs (e.g., rotated) by checking thumb is away from hand center
        center_x = (min_x + max_x) / 2
        if abs(thumb_tip[0] - center_x) > 0.3 * box_w:
            return '👍', 'Thumbs Up'

    # OK sign -> thumb and index tips close while other fingers may be extended or relaxed
    idx_tip = pts[8]
    thumb_tip = pts[4]
    tip_dist = distance(idx_tip, thumb_tip)
    # If thumb and index tip are close relative to hand size, it's an OK sign
    if tip_dist < 0.25 * diag:
        # ensure it's not just a tiny pinch by checking some minimal separation
        if tip_dist > 8:
            return '👌', 'OK'

    # If none match, return None
    return None, 'Unknown'


# ---------- Drawing helper: overlay emoji text or render with PIL if available ----------

def overlay_emoji_on_frame(frame, emoji, center_xy, size=3.0):
    """
    Overlay the emoji (a Unicode char such as '👍') on the frame centered at center_xy.
    If Pillow is available, uses a truetype emoji-capable font for better rendering.
    Falls back to cv2.putText (which often can't render colored emoji).
    frame: BGR OpenCV image (numpy array)
    emoji: str
    center_xy: (x, y) pixel coordinates where emoji center should be placed
    size: float scale factor for emoji font
    """
    x, y = center_xy
    h, w, _ = frame.shape
    if PIL_AVAILABLE:
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Try to find emoji-capable font
        font = None
        for p in EMOJI_FONT_PATHS:
            try:
                font = ImageFont.truetype(p, int(64 * size))
                break
            except Exception:
                font = None
        if font is None:
            try:
                # fallback to default PIL font (may not produce colored emoji)
                font = ImageFont.load_default()
            except Exception:
                font = None

        # Get size of the emoji text in a robust way across Pillow versions
        text_w = text_h = None
        if font:
            # Preferred: use font.getsize if available
            try:
                text_w, text_h = font.getsize(emoji)
            except Exception:
                text_w = text_h = None
            # Fallback: use draw.textbbox if available (newer Pillow)
            if text_w is None:
                try:
                    bbox = draw.textbbox((0, 0), emoji, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w = text_h = None
        if text_w is None:
            # Final fallback: estimate size from requested font size
            est = int(64 * size)
            text_w, text_h = est, est

        # Position such that text center aligns with center_xy
        pos = (int(x - text_w / 2), int(y - text_h / 2))
        # Draw text - try black outline for readability
        outline_color = "black"
        # Draw simple outline by drawing text multiple times offset by 1 pixel
        if font:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((pos[0] + dx, pos[1] + dy), emoji, font=font, fill=outline_color)
            draw.text(pos, emoji, font=font, fill="white")
        else:
            draw.text(pos, emoji, fill="white")

        # Convert back to OpenCV BGR
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # Fallback: draw white text using OpenCV (may not support emoji glyphs)
        # We'll draw a filled circle background for readability and then text label below.
        radius = int(40 * size)
        # Background circle
        cv2.circle(frame, (x, y), radius, (50, 50, 50), -1)  # dark circle
        # Emoji / label
        try:
            # Attempt to draw the emoji character directly; may show as square on some systems
            cv2.putText(frame, emoji, (x - radius + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        except Exception:
            # Final fallback: write the name (ASCII)
            cv2.putText(frame, "EMOJI", (x - radius + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, size * 0.6, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


# ---------- Main webcam loop ----------

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Configure MediaPipe Hands:
    # - static_image_mode=False (for video)
    # - max_num_hands=1 (the requirement: detect one hand)
    # - min_detection_confidence / min_tracking_confidence set to reasonable defaults
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    emoji_to_display = None
    label_text = ''
    last_detection_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape

            # Convert the BGR image to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture_emoji = None
            gesture_name = 'No hand'

            if results.multi_hand_landmarks and results.multi_handedness:
                # We process only the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_handedness = results.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

                # Draw hand landmarks on the frame for debugging/visual feedback
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                # Detect gesture
                emoji_char, name = detect_gesture(hand_landmarks.landmark, hand_handedness, img_w, img_h)
                if emoji_char:
                    gesture_emoji = emoji_char
                    gesture_name = name
                    last_detection_time = time.time()
                else:
                    # Keep the previous emoji shown briefly to avoid flicker; show "Unknown" label
                    gesture_emoji = None
                    gesture_name = 'Unknown'

                # Compute bounding box center for overlay placement
                xs = [int(lm.x * img_w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * img_h) for lm in hand_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                center_x = int((min_x + max_x) / 2)
                center_y = int(min_y - 20) if min_y - 20 > 20 else int(min_y + (max_y - min_y) // 2)

                # If an emoji is recognized, overlay it at the hand location
                if gesture_emoji:
                    overlay_emoji_on_frame(frame, gesture_emoji, (center_x, center_y), size=2.2)
                    emoji_to_display = gesture_emoji
                    label_text = gesture_name
                else:
                    # If unknown but recent detection existed, show last emoji for small grace period
                    if emoji_to_display and (time.time() - last_detection_time) < 0.4:
                        overlay_emoji_on_frame(frame, emoji_to_display, (center_x, center_y), size=2.2)

                # Draw bounding box around hand
                cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (255, 200, 0), 2)
                # Draw label
                cv2.putText(frame, f"{hand_handedness} hand: {gesture_name}", (min_x, min_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # No hand detected; clear emoji after short timeout
                if emoji_to_display and (time.time() - last_detection_time) > 0.6:
                    emoji_to_display = None
                    label_text = ''

            # FPS calculation and display
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 1e-6 else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show instruction to quit
            cv2.putText(frame, "Press 'q' to quit", (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Show the frame
            cv2.imshow("Hand Emoji Detector", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
