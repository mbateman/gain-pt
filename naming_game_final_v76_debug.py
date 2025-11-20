# naming_game_final_v76_debug.py
import streamlit as st
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).parent
NAMINGGAME_DIR = BASE_DIR / "diagrams" / "NamingGame"
SCENARIOS_DIR = BASE_DIR / "diagrams" / "Scenarios"
BOARD_IMAGE_PATH = NAMINGGAME_DIR / "NamingGameBoard.png"

# ======================================================
# VERIFIED 74-SLOT BOARD PATTERN
# ======================================================
BOARD_SLOTS = [
    'start', 'guess', 'sentence', 'naming', 'sentence', 'guess', 'naming',
    'naming', 'naming', 'sentence', 'naming', 'naming', 'stop', 'sentence',
    'naming', 'guess', 'naming', 'guess', 'naming', 'sentence', 'moveahead',
    'naming', 'stop', 'sentence', 'naming', 'guess', 'naming', 'moveahead',
    'sentence', 'naming', 'guess', 'moveback', 'naming', 'guess', 'naming',
    'sentence', 'guess', 'moveforward', 'sentence', 'naming', 'guess', 'naming',
    'sentence', 'guess', 'naming', 'sentence', 'naming', 'guess', 'naming',
    'naming', 'guess', 'moveback', 'naming', 'naming', 'sentence', 'naming',
    'naming', 'guess', 'naming', 'stop', 'naming', 'guess', 'moveahead',
    'sentence', 'guess', 'moveback', 'naming', 'guess', 'naming', 'sentence',
    'naming', 'guess', 'sentence', 'end'
]
TOTAL_SLOTS = len(BOARD_SLOTS)

# ======================================================
# BOARD GEOMETRY
# ======================================================
_base_board = Image.open(BOARD_IMAGE_PATH).convert("RGBA")
BOARD_W, BOARD_H = _base_board.size

from math import atan2, cos, sin, pi

def compute_perimeter_coords(width, height, total_slots, margin=48):
    """
    Compute coordinates along the perimeter.
    We'll correct placement later by moving tokens toward the center.
    """
    coords = []
    left, right, top, bottom = margin, width - margin, margin, height - margin
    perim = 2 * ((right - left) + (bottom - top))

    for i in range(total_slots):
        frac = i / total_slots
        dist = frac * perim
        bottom_len = right - left
        left_len = bottom - top
        top_len = right - left

        if dist <= bottom_len:
            x = right - dist
            y = bottom
        elif dist <= bottom_len + left_len:
            d = dist - bottom_len
            x = left
            y = bottom - d
        elif dist <= bottom_len + left_len + top_len:
            d = dist - bottom_len - left_len
            x = left + d
            y = top
        else:
            d = dist - bottom_len - left_len - top_len
            x = right
            y = top + d
        coords.append((x, y))
    return coords

COORDS = compute_perimeter_coords(BOARD_W, BOARD_H, TOTAL_SLOTS, margin=48)

# --- helper: compute visual centroids for each slot once and cache in session_state ---
def compute_slot_centroids(base_img, coords, crop_radius=30, threshold=30):
    """
    For each (x,y) in coords, crop a square patch of size (2*crop_radius+1).
    Convert to grayscale, compute a mask of 'foreground' pixels using adaptive thresholding
    relative to patch median, then compute the centroid of the largest connected region
    (or weighted centroid of foreground). Returns list of (x_abs, y_abs) centroids.
    """
    if "slot_centroids" in st.session_state and st.session_state["slot_centroids"]:
        return st.session_state["slot_centroids"]

    im = base_img.convert("L")  # grayscale
    W, H = im.size
    arr = np.array(im)

    centroids = []
    for (cx, cy) in coords:
        # integer crop extents
        left = int(max(0, cx - crop_radius))
        top = int(max(0, cy - crop_radius))
        right = int(min(W, cx + crop_radius + 1))
        bottom = int(min(H, cy + crop_radius + 1))

        patch = arr[top:bottom, left:right]
        if patch.size == 0:
            centroids.append((cx, cy))
            continue

        # robust local threshold: background ~ median; foreground darker/lighter -> use abs diff
        med = np.median(patch)
        # compute absolute deviation from median
        dev = np.abs(patch.astype(np.int16) - int(med))
        # mask pixels that differ sufficiently from local median
        mask = dev > threshold

        # if mask is too sparse, try lower threshold
        if mask.sum() < 5:
            mask = dev > (threshold // 2)

        # compute weighted centroid using mask (if any)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            # fallback: try simple edge detection via Laplacian-like filter
            patch_f = patch.astype(np.float32)
            gy, gx = np.gradient(patch_f)
            grad = np.hypot(gx, gy)
            # threshold gradient
            gmask = grad > (np.percentile(grad, 90))
            ys, xs = np.nonzero(gmask)

        if len(xs) == 0:
            # give up for this slot: use original coord
            centroids.append((cx, cy))
            continue

        # compute centroid in patch coordinates
        x_mean = xs.mean()
        y_mean = ys.mean()
        # convert back to board coordinates
        x_abs = left + x_mean
        y_abs = top + y_mean
        centroids.append((float(x_abs), float(y_abs)))

    # cache in session state
    st.session_state["slot_centroids"] = centroids
    return centroids


# --- manual fine-tuning offsets for token placement ---
TOKEN_OFFSET_X = +18   # move token 18px to the left
TOKEN_OFFSET_Y = +12   # move token 10px downward

# START_ARROW_POS = (COORDS[0][0] + 15, COORDS[0][1] + 20)  # slightly down/right of first slot

# --------------------------
# VISUAL MAP CREATION
# --------------------------
N = len(COORDS)  # should be 74
START_ARROW_POS = COORDS[0]  # index 0 = start arrow

# Apply fine alignment offset
START_ARROW_POS = (START_ARROW_POS[0] + TOKEN_OFFSET_X,
                   START_ARROW_POS[1] + TOKEN_OFFSET_Y)

# visual_map = COORDS[:]  # direct mapping, no rotation
# TOTAL_SLOTS = len(visual_map)


def draw_token_on_board(base_img, position_index, coords, show_indices=False, token_color="red"):
    board = base_img.copy()
    draw = ImageDraw.Draw(board)

    if show_indices:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        except Exception:
            font = None
        for i, (x, y) in enumerate(coords):
            draw.text((x, y - 10), str(i), fill="black", font=font)

    # Clamp position
    idx = max(0, min(position_index, len(coords) - 1))
    x, y = coords[idx]

    # Apply alignment offset
    x += TOKEN_OFFSET_X
    y += TOKEN_OFFSET_Y

    r = max(10, int(min(base_img.size) * 0.015))
    draw.ellipse((x - r, y - r, x + r, y + r), fill=token_color, outline="black")
    return board



# ======================================================
# SCENARIO IMAGES
# ======================================================
EXCLUDE = {"start.png", "finish.png", "chance.png"}
EXCLUDE_SUBSTR = {"board.png"}

def load_scenario_images():
    imgs = []
    for sub in SCENARIOS_DIR.iterdir():
        if sub.is_dir():
            for f in sub.glob("*.png"):
                name = f.name.lower()
                if name not in EXCLUDE and not any(x in name for x in EXCLUDE_SUBSTR):
                    imgs.append(f)
    return imgs

SCENARIO_IMAGES = load_scenario_images()

def random_image():
    if not SCENARIO_IMAGES:
        return None, ""
    img = random.choice(SCENARIO_IMAGES)
    word = img.stem.replace("_", " ").lower()
    return img, word

def is_correct(user_input, correct_word):
    if not user_input:
        return False
    ui = user_input.strip().lower()
    return correct_word in ui or ui in correct_word

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "position": 0,
    "roll": None,
    "task": "start",
    "correct_word": "",
    "img_path": None,
    "show_indices": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# STREAMLIT UI
# ======================================================
st.title("🧠 GAIN-PT: Naming Game (Final 74-slot layout + Debug Overlay)")
st.markdown("Players move clockwise from the start arrow (index 0) to the END slot (index 73).")

col_controls, col_board = st.columns([1, 2])

# ======================================================
# GAME CONTROLS
# ======================================================
with col_controls:
    if st.session_state.position >= TOTAL_SLOTS - 1:
        st.success("🏁 Game complete! You reached END.")
    else:
        if st.button("🎲 Roll Dice"):
            st.session_state.roll = random.randint(1, 6)
            st.session_state.position = min(
                st.session_state.position + st.session_state.roll, TOTAL_SLOTS - 1
            )
            st.session_state.task = BOARD_SLOTS[st.session_state.position]
            if st.session_state.task not in ["start", "end"]:
                st.session_state.img_path, st.session_state.correct_word = random_image()

    st.markdown(f"**Position:** {st.session_state.position} / {TOTAL_SLOTS - 1}")
    if st.session_state.roll:
        st.markdown(f"**You rolled:** {st.session_state.roll}")
    st.markdown(f"**Slot type:** `{st.session_state.task}`")

    if st.button("🔁 Restart"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    # Debug toggle
    st.session_state.show_indices = st.checkbox("🧩 Show slot indices (debug)", value=st.session_state.show_indices)

# ======================================================
# BOARD DISPLAY
# ======================================================
with col_board:
    board_img = draw_token_on_board(_base_board, st.session_state.position, COORDS, show_indices=st.session_state.show_indices)
    # board_img = draw_token_on_board(_base_board, st.session_state.position, COORDS, show_indices=st.session_state["show_indices"])

    st.image(board_img, caption="Board with token", width='stretch')

# ======================================================
# TASK HANDLING
# ======================================================
slot = st.session_state.task

def ensure_image():
    if not st.session_state.get("img_path") or not st.session_state.get("correct_word"):
        st.session_state.img_path, st.session_state.correct_word = random_image()

if slot == "start":
    st.info("🎯 Starting position — roll to begin!")
elif slot == "end":
    st.success("🏁 You've reached the END. Great work!")
elif slot == "naming":
    ensure_image()
    st.image(st.session_state.img_path, width=220)
    ans = st.text_input("Name this object:", key="n_input")
    if st.button("Submit naming"):
        if is_correct(ans, st.session_state.correct_word):
            st.success("✅ Correct!")
        else:
            st.error(f"❌ The answer was {st.session_state.correct_word}.")
elif slot == "sentence":
    ensure_image()
    st.image(st.session_state.img_path, width=220)
    sent = st.text_area("Write a sentence with this word:", key="s_input")
    if st.button("Submit sentence"):
        if is_correct(sent, st.session_state.correct_word):
            st.success("✅ Sentence includes target word.")
        else:
            st.warning(f"⚠️ Please include {st.session_state.correct_word}.")
elif slot == "guess":
    ensure_image()
    correct_img = st.session_state.img_path
    correct_word = st.session_state.correct_word
    distractors = random.sample(
        [img for img in SCENARIO_IMAGES if img != correct_img], 2
    )
    options = random.sample(distractors + [correct_img], 3)
    choice = st.radio(
        f"Select the picture for '{correct_word}':",
        options,
        format_func=lambda p: p.stem,
        key="g_choice",
    )
    st.image(choice, width=200)
    if st.button("Submit guess"):
        if choice == correct_img:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ The correct answer was {correct_word}.")
elif slot == "moveahead":
    st.success("🎉 Move ahead 3 spaces!")
    st.session_state.position = min(st.session_state.position + 3, TOTAL_SLOTS - 1)
    st.session_state.task = BOARD_SLOTS[st.session_state.position]
elif slot == "moveback":
    st.warning("↩️ Move back 3 spaces!")
    st.session_state.position = max(st.session_state.position - 3, 0)
    st.session_state.task = BOARD_SLOTS[st.session_state.position]
elif slot == "stop":
    st.info("🛑 Stop — skip next turn (not implemented in single-player).")
else:
    st.write(f"Slot: {slot}")
