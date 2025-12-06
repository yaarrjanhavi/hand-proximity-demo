import cv2
import numpy as np

# ---- Parameters you can tweak ----
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Virtual object (a rectangle) position
obj_x1, obj_y1 = 400, 150
obj_x2, obj_y2 = 550, 300
obj_center = ((obj_x1 + obj_x2) // 2, (obj_y1 + obj_y2) // 2)

# Distance thresholds (in pixels)
WARNING_DIST = 200
DANGER_DIST = 80

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def get_hand_point(frame):
    # Convert to HSV for simple skin mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rough skin-color range (you may need to tune these for your lighting / skin tone)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Noise removal
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    # Largest contour assumed to be hand
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 2000:
        # Too small, probably noise
        return None, mask

    # Contour centroid as hand point
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural interaction

    hand_point, mask = get_hand_point(frame)

    # Draw virtual object
    cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2),
                  (255, 0, 0), 2)
    cv2.circle(frame, obj_center, 4, (255, 0, 0), -1)

    state_text = "SAFE"
    state_color = (0, 255, 0)
    danger_warning_text = ""

    if hand_point is not None:
        hx, hy = hand_point

        # Draw hand point
        cv2.circle(frame, (hx, hy), 8, (0, 255, 255), -1)

        # Distance to object center
        dx = hx - obj_center[0]
        dy = hy - obj_center[1]
        dist = (dx * dx + dy * dy) ** 0.5

        # Draw line from hand to object center
        cv2.line(frame, (hx, hy), obj_center, (0, 255, 255), 2)

        # State logic
        if dist <= DANGER_DIST:
            state_text = "DANGER"
            state_color = (0, 0, 255)
            danger_warning_text = "DANGER DANGER"
        elif dist <= WARNING_DIST:
            state_text = "WARNING"
            state_color = (0, 255, 255)
        else:
            state_text = "SAFE"
            state_color = (0, 255, 0)

        # Debug: show distance
        cv2.putText(frame, f"dist: {int(dist)}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    # Overlay current state
    cv2.putText(frame, f"STATE: {state_text}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                state_color, 3)

    if danger_warning_text:
        cv2.putText(frame, danger_warning_text,
                    (100, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 3)

    cv2.imshow("Hand Proximity Demo", frame)

    # Optional: show mask window for tuning
    # cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
