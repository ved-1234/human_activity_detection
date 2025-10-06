import cv2
import numpy as np

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def safe_point(p):
    """Ensure keypoint is valid and converted to int tuple."""
    if p is None:
        return None
    # Some keypoints are tensors like tensor([x,y,score])
    if hasattr(p, "detach"):
        p = p.cpu().numpy()
    # If it has score, take only first two coords
    if len(p) >= 2:
        return (int(p[0]), int(p[1]))
    return None


def draw_line(image, p1, p2, color):
    p1 = safe_point(p1)
    p2 = safe_point(p2)
    if p1 is not None and p2 is not None:
        cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def draw_circle(image, p, color=WHITE_COLOR, radius=4):
    p = safe_point(p)
    if p is not None:
        cv2.circle(image, p, radius, color, -1)


def find_person_indicies(scores):
    return [i for i, s in enumerate(scores) if s > 0.9]


def filter_persons(outputs):
    persons = {}
    p_indicies = find_person_indicies(outputs["instances"].scores)
    for x in p_indicies:
        desired_kp = outputs["instances"].pred_keypoints[x][:].to("cpu")
        persons[x] = desired_kp
    return (persons, p_indicies)


def draw_keypoints(person, img):
    # unpack all keypoints
    nose, l_eye, r_eye, l_ear, r_ear, \
    l_shoulder, r_shoulder, l_elbow, r_elbow, \
    l_wrist, r_wrist, l_hip, r_hip, \
    l_knee, r_knee, l_ankle, r_ankle = person

    # Draw skeleton
    draw_line(img, l_shoulder, l_elbow, GREEN_COLOR)
    draw_line(img, l_elbow, l_wrist, GREEN_COLOR)
    draw_line(img, l_shoulder, r_shoulder, GREEN_COLOR)
    draw_line(img, l_shoulder, l_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_elbow, GREEN_COLOR)
    draw_line(img, r_elbow, r_wrist, GREEN_COLOR)
    draw_line(img, l_hip, r_hip, GREEN_COLOR)
    draw_line(img, l_hip, l_knee, GREEN_COLOR)
    draw_line(img, l_knee, l_ankle, GREEN_COLOR)
    draw_line(img, r_hip, r_knee, GREEN_COLOR)
    draw_line(img, r_knee, r_ankle, GREEN_COLOR)

    # Draw keypoint circles
    for kp in [l_eye, r_eye, l_wrist, r_wrist,
               l_shoulder, r_shoulder, l_elbow, r_elbow,
               l_hip, r_hip, l_knee, r_knee,
               l_ankle, r_ankle]:
        draw_circle(img, kp)
