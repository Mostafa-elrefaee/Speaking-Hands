import cv2
import time
from pathlib import Path

# ----------------------------
# Settings
# ----------------------------
OUTPUT_DIR = Path("videos/finish")
OUTPUT_FILE = OUTPUT_DIR / "fn12.mp4"

DURATION_SECONDS = 2
CAMERA_INDEX = 0

# ----------------------------
# Prepare output directory
# ----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam.")

# Try to keep the webcam's native resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Some webcams return 0/unknown FPS.
if fps <= 0 or fps != fps:
    fps = 30.0

# MP4 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_FILE),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

print(f"Recording for {DURATION_SECONDS} seconds...")
print(f"Saving to: {OUTPUT_FILE.resolve()}")

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read a frame from the webcam.")
        break

    elapsed = time.time() - start_time

    # Show countdown/progress
    remaining = max(0, DURATION_SECONDS - elapsed)

    cv2.putText(
        frame,
        f"Recording: {remaining:.1f}s",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)
    cv2.imshow("Recording", frame)

    # Stop after the requested duration
    if elapsed >= DURATION_SECONDS:
        break

    # Emergency stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Done. Video saved to: {OUTPUT_FILE.resolve()}")