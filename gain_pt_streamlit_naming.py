# gain_pt_streamlit_naming.py
"""
GAIN Naming Game v7.2

Improvements:
- Per-side coordinate generation using explicit slot counts per edge:
    bottom=23 (0..22), left=15 (23..37), top=22 (38..59), right=14 (60..73)
- Calibration helper: show (index,x,y), capture corrected coordinates, export CSV
- CSV corrections merge + preview overlay (red base, green corrected), token always visible
- Streamlit 1.50.1 compatibility (st.rerun(), width="stretch"/"content")
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
import random
import math

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="GAIN Naming Game v7.2", layout="wide")
DIAGRAMS_DIR = Path("diagrams")
BOARD_PATH = DIAGRAMS_DIR / "NamingGame" / "NamingGameBoard.png"
SCENARIOS_DIR = DIAGRAMS_DIR / "Scenarios"
TOTAL_SLOTS = 74
COORDS_BASE_FILE = Path("coords_base.json")
COORDS_CORRECTED_FILE = Path("coords_corrected.json")

DEFAULT_TOKEN_OFFSET_X = 0
DEFAULT_TOKEN_OFFSET_Y = 0

# -----------------------
# Base pattern (74 slots)
# -----------------------
BASE_PATTERN = [
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
assert len(BASE_PATTERN) == TOTAL_SLOTS

# -----------------------
# Utility functions
# -----------------------
def load_board_image():
    if not BOARD_PATH.exists():
        st.error(f"Board image not found at {BOARD_PATH}. Place it at {BOARD_PATH}")
        st.stop()
    return Image.open(BOARD_PATH).convert("RGBA")

def compute_coords_by_side(img_w, img_h,
                           bottom_count=23, left_count=15, top_count=22, right_count=14,
                           inner_margin=60, outer_margin=40):
    """
    Compute coordinates for each slot grouped by side according to counts provided.
    Coordinates are placed along a rectangle inset by inner_margin, but we create
    separate line segments for each side and evenly distribute slots along them.

    - bottom_count covers indices 0..bottom_count-1 (leftwards along bottom edge, from start arrow)
      (We will arrange so index 0 is the start arrow at right-bottom)
    - left_count covers next indices (upwards)
    - top_count covers across top (left -> right)
    - right_count covers downwards along right side (top -> bottom)
    """
    # Inset rectangle for slot midpoints
    left = inner_margin
    right = img_w - inner_margin
    top = inner_margin
    bottom = img_h - inner_margin

    coords = []

    # Bottom side: we want to go from right -> left (start arrow is at right-bottom)
    # positions evenly spaced including both ends when appropriate.
    if bottom_count > 1:
        xs = [right - i * (right - left) / (bottom_count - 1) for i in range(bottom_count)]
    else:
        xs = [(left + right) / 2.0]
    ys = [bottom] * bottom_count
    coords.extend(list(zip(xs, ys)))  # indices 0 .. bottom_count-1

    # Left side: go from bottom -> top (upwards)
    if left_count > 1:
        ys_left = [bottom - i * (bottom - top) / (left_count - 1) for i in range(left_count)]
    else:
        ys_left = [(top + bottom) / 2.0]
    xs_left = [left] * left_count
    coords.extend(list(zip(xs_left, ys_left)))

    # Top side: go from left -> right
    if top_count > 1:
        xs_top = [left + i * (right - left) / (top_count - 1) for i in range(top_count)]
    else:
        xs_top = [(left + right) / 2.0]
    ys_top = [top] * top_count
    coords.extend(list(zip(xs_top, ys_top)))

    # Right side: go from top -> bottom
    if right_count > 1:
        ys_right = [top + i * (bottom - top) / (right_count - 1) for i in range(right_count)]
    else:
        ys_right = [(top + bottom) / 2.0]
    xs_right = [right] * right_count
    coords.extend(list(zip(xs_right, ys_right)))

    # Sanity: length should be bottom+left+top+right == TOTAL_SLOTS
    expected = bottom_count + left_count + top_count + right_count
    if expected != TOTAL_SLOTS:
        st.warning(f"Side counts sum {expected} != TOTAL_SLOTS {TOTAL_SLOTS}. Adjust counts.")
    # Convert to floats and return
    coords = [(float(x), float(y)) for x, y in coords]
    return coords

# -----------------------
# Base coords loader/generator (per-side)
# -----------------------
def make_base_coords_by_side():
    board = load_board_image()
    W, H = board.size
    # Use the side counts you specified
    coords = compute_coords_by_side(W, H, bottom_count=23, left_count=15, top_count=22, right_count=14, inner_margin=60)
    # Save base coords
    with open(COORDS_BASE_FILE, "w") as f:
        json.dump(coords, f)
    return coords

def load_base_coords():
    if COORDS_BASE_FILE.exists():
        with open(COORDS_BASE_FILE, "r") as f:
            c = json.load(f)
        if len(c) == TOTAL_SLOTS:
            return c
    return make_base_coords_by_side()

def load_corrected_coords():
    base = load_base_coords()
    if COORDS_CORRECTED_FILE.exists():
        with open(COORDS_CORRECTED_FILE, "r") as f:
            c = json.load(f)
        if len(c) == TOTAL_SLOTS:
            return c
    return base

def merge_with_csv(base_coords, df):
    coords = list(base_coords)
    for _, r in df.iterrows():
        i = int(r["index"])
        x = float(r["x"])
        y = float(r["y"])
        if 0 <= i < len(coords):
            coords[i] = (x, y)
    return coords

# -----------------------
# Scenario images for tasks
# -----------------------
def load_scenario_images():
    imgs = []
    if SCENARIOS_DIR.exists():
        for sub in SCENARIOS_DIR.iterdir():
            if sub.is_dir():
                for f in sorted(sub.glob("*.png")):
                    n = f.name.lower()
                    if n not in {"start.png", "finish.png", "chance.png"} and "board" not in n:
                        imgs.append(f)
    return imgs

SCENARIO_IMAGES = load_scenario_images()

def random_task_image_and_word():
    if not SCENARIO_IMAGES:
        return None, ""
    img = random.choice(SCENARIO_IMAGES)
    return img, img.stem.replace("_", " ").lower()

def is_correct_answer(user_text, target_word):
    if not user_text:
        return False
    ui = user_text.strip().lower()
    return target_word in ui or ui in target_word

# -----------------------
# Session defaults
# -----------------------
st.session_state.setdefault("position", 0)
st.session_state.setdefault("roll", None)
st.session_state.setdefault("task_img", None)
st.session_state.setdefault("task_word", "")
st.session_state.setdefault("token_offset_x", DEFAULT_TOKEN_OFFSET_X)
st.session_state.setdefault("token_offset_y", DEFAULT_TOKEN_OFFSET_Y)
st.session_state.setdefault("captured_corrections", [])  # list of dicts {index,x,y}
st.session_state.setdefault("last_task_position", None)


# -----------------------
# Sidebar: calibration controls
# -----------------------
st.sidebar.header("Calibration & overlay")

show_indices = st.sidebar.checkbox("Show COORD indices (0–73)", value=False)
show_overlay = st.sidebar.checkbox("Show base vs corrected overlay", value=False)
show_coord_grid = st.sidebar.checkbox("Show coordinate grid", value=False)
show_fine_tune = st.sidebar.checkbox("Show fine tune", value=False)
roll_one = st.sidebar.checkbox("Roll dice one position forward", value=False)

uploaded = st.sidebar.file_uploader("Upload CSV corrections (index,x,y)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if {"index", "x", "y"}.issubset(df.columns):
            base_coords = load_base_coords()
            merged = merge_with_csv(base_coords, df)
            with open(COORDS_CORRECTED_FILE, "w") as f:
                json.dump(merged, f)
            st.sidebar.success(f"Merged {len(df)} corrections → saved to {COORDS_CORRECTED_FILE.name}")
        else:
            st.sidebar.error("CSV must include columns: index,x,y")
    except Exception as e:
        st.sidebar.error(f"CSV load error: {e}")

if st.sidebar.button("💾 Save current merged coords"):
    merged = load_corrected_coords()
    with open(COORDS_CORRECTED_FILE, "w") as f:
        json.dump(merged, f)
    st.sidebar.success("Saved coords_corrected.json")

st.sidebar.markdown("---")
st.sidebar.markdown("Create CSV: index,x,y (only rows you want to correct).")

# -----------------------
# Load coords
# -----------------------
base_coords = load_base_coords()
corrected_coords = load_corrected_coords()
active_coords = corrected_coords  # used for drawing

# -----------------------
# Drawing & overlay helpers
# -----------------------
def draw_board_with_overlays(position=None, show_indices=False, show_overlay=False):
    board = load_board_image().copy()
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    if show_overlay:
        # show base coords in red (slightly left-up), corrected in green (slightly right-down)
        for i, (x, y) in enumerate(base_coords):
            draw.text((x - 14, y - 14), str(i), fill="red", font=font)
        for i, (x, y) in enumerate(corrected_coords):
            draw.text((x + 10, y + 10), str(i), fill="green", font=font)
    elif show_indices:
        index_offset_x = 14
        index_offset_y = -14
        for i, (x, y) in enumerate(active_coords):
            draw.text((x - index_offset_x, y - index_offset_y), str(i), fill="black", font=font)

    # always draw token
    if position is not None:
        x, y = active_coords[position]
        x += st.session_state["token_offset_x"]
        y += st.session_state["token_offset_y"]
        r = max(8, int(min(board.size) * 0.015))
        draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="black")

    return board

def draw_coordinate_grid(img, step=25, color=(0, 0, 0, 80)):
    """
    Overlay a semi-transparent coordinate grid on the board image.
    step = pixel spacing between grid lines.
    color = RGBA for grid lines.
    """
    grid = img.copy()
    draw = ImageDraw.Draw(grid, "RGBA")
    W, H = grid.size

    # interval = 10
    # vertical lines
    for x in range(0, W, step):
        draw.line((x, 0, x, H), fill=color, width=1)
        draw.text((x + 2, 2), str(x), fill=color)

    # horizontal lines
    for y in range(0, H, step):
        draw.line((0, y, W, y), fill=color, width=1)
        draw.text((2, y + 2), str(y), fill=color)

    return grid


# -----------------------
# Calibration helper: show current slot coords and capture
# -----------------------
def get_current_slot_info():
    idx = int(st.session_state["position"])
    slot_type = BASE_PATTERN[idx]
    x, y = active_coords[idx]
    # present pixel ints
    return {"index": idx, "slot_type": slot_type, "x": int(round(x)), "y": int(round(y))}

def capture_current_coordinate():
    info = get_current_slot_info()
    # append to captured list, avoid duplicates (replace if same index exists)
    cc = st.session_state["captured_corrections"]
    # remove existing for this index
    token_offset_x = st.session_state["token_offset_x"]
    token_offset_y = st.session_state["token_offset_y"]
    cc = [c for c in cc if c["index"] != info["index"]]
    cc.append({"index": info["index"], "x": info["x"]+token_offset_x, "y": info["y"]+token_offset_y})
    st.session_state["captured_corrections"] = sorted(cc, key=lambda r: r["index"])
    st.success(f"Captured index {info['index']} → ({info['x']+token_offset_x}, {info['y']+token_offset_y})")

def download_captured_csv():
    cc = st.session_state["captured_corrections"]
    if not cc:
        st.warning("No captured corrections to download.")
        return None
    df = pd.DataFrame(cc)[["index","x","y"]]
    csv = df.to_csv(index=False)
    return csv

# -----------------------
# Main UI
# -----------------------
st.title("GAIN Naming Game — v7.2 (per-side coords + calibration helper)")
st.markdown("Start index 0 = start arrow. Index 73 = END. Use sidebar for overlay and CSV upload.")

col_controls, col_board = st.columns([1, 2])

with col_board:
    board_img = draw_board_with_overlays(position=st.session_state["position"],
                                         show_indices=show_indices,
                                         show_overlay=show_overlay)
    
    # ⬇ NEW: coordinate grid overlay toggle
    if show_coord_grid:         
        board_img = draw_coordinate_grid(board_img)
        
    st.image(board_img, width="stretch")

with col_controls:
    st.session_state.setdefault("last_task_position", None)
    st.markdown("### Controls")
    if st.button("🎲 Roll Dice"):
        if roll_one:
            roll = 1    
        else:
            roll = random.randint(1, 6)
        st.session_state["roll"] = roll
        st.session_state["position"] = min(st.session_state["position"] + roll, TOTAL_SLOTS - 1)
        # prepare task for playable slots
        slot_type = BASE_PATTERN[st.session_state["position"]]
        # Only assign a new task if entering a playable slot *for the first time*
        if slot_type in {"naming", "sentence", "guess"}:
            if st.session_state["last_task_position"] != st.session_state["position"]:
                st.session_state["task_img"], st.session_state["task_word"] = random_task_image_and_word()
                st.session_state["last_task_position"] = st.session_state["position"]
        else:
            st.session_state["task_img"], st.session_state["task_word"] = None, ""
            st.session_state["last_task_position"] = None
        st.rerun()
    if st.button("🔁 Restart"):
        st.session_state.update({"position": 0, "roll": None, "task_img": None, "task_word": ""})
        st.rerun()

    st.markdown(f"**Position:** {st.session_state['position']} / {TOTAL_SLOTS - 1}")
    st.markdown(f"**Slot type:** {BASE_PATTERN[st.session_state['position']]}")
    if st.session_state.get("roll") is not None:
        st.markdown(f"**Last roll:** {st.session_state['roll']}")

    if show_fine_tune:
        st.markdown("---")
        st.markdown("### Token fine-tune")
        st.session_state["token_offset_x"] = st.slider("Offset X (px)", -80, 80, value=st.session_state["token_offset_x"], step=1)
        st.session_state["token_offset_y"] = st.slider("Offset Y (px)", -80, 80, value=st.session_state["token_offset_y"], step=1)
        # No explicit apply button — sliders update immediately

        st.markdown("---")
        st.markdown("### Calibration helper")
        curr = get_current_slot_info()
        st.write(f"Index: **{curr['index']}**")
        st.write(f"Slot type: **{curr['slot_type']}**")
        st.write(f"Active coords: **{curr['x']}, {curr['y']}**")
        token_offset_x = st.session_state["token_offset_x"]
        token_offset_y = st.session_state["token_offset_y"]
        st.write(f"Adjusted coords: **{curr['x']+token_offset_x}, {curr['y']+token_offset_y}**")
        if st.button("📍 Capture coordinate for this index"):
            capture_current_coordinate()
        captured = st.session_state["captured_corrections"]
        if captured:
            df_cap = pd.DataFrame(captured)[["index","x","y"]]
            st.write("Captured corrections (you can download and re-upload as CSV):")
            st.dataframe(df_cap)
            csv_data = download_captured_csv()
            if csv_data:
                st.download_button("⬇️ Download captured corrections CSV", data=csv_data, file_name="captured_corrections.csv", mime="text/csv")
        else:
            st.info("No captured corrections yet. Capture the currently displayed token coordinate when aligned.")

# -----------------------
# Task pane
# -----------------------
slot = BASE_PATTERN[st.session_state["position"]]
st.markdown("---")
if slot == "start":
    st.info("🎯 Start: on arrow — roll to begin.")
elif slot == "end":
    st.success("🏁 END reached — finished the circuit!")
elif slot == "naming":
    st.subheader("Naming")
    if st.session_state.get("task_img"):
        st.image(st.session_state["task_img"], width=250)
    ans = st.text_input("What is this called?", key="naming_input")
    if st.button("Submit naming"):
        if is_correct_answer(ans, st.session_state.get("task_word", "")):
            st.success("✅ Correct")
        else:
            st.error(f"❌ Expected: {st.session_state.get('task_word','')}")
elif slot == "sentence":
    st.subheader("Sentence")
    if st.session_state.get("task_img"):
        st.image(st.session_state["task_img"], width=250)
    sent = st.text_area("Write a sentence using the target word", key="sentence_input")
    if st.button("Submit sentence"):
        if is_correct_answer(sent, st.session_state.get("task_word", "")):
            st.success("✅ Contains target")
        else:
            st.warning(f"⚠️ Try to include: {st.session_state.get('task_word','')}")

# ------------------------------
# Corrected Guess Logic (v7.2 stable)
# ------------------------------
elif slot == "guess":
 
    st.subheader("Guess Task")

    # 1. Initialise the guess state ONCE per slot entry
    if "guess_options" not in st.session_state:

        correct_img = st.session_state.get("task_img")
        correct_key = correct_img.as_posix()
        
        # pick distractors once
        distractors = [p for p in SCENARIO_IMAGES if p != correct_img]
        distractors = random.sample(distractors, min(4, len(distractors)))

        options = [
            {"label": correct_img.stem, "path": correct_img, "key": correct_key, "is_correct": True}
        ]
        for d in distractors:
            options.append({
                "label": d.stem,
                "path": d,
                "key": d.as_posix(),
                "is_correct": False
            })

        random.shuffle(options)

        st.session_state["guess_options"] = options
        labels = [o["label"].replace("_", " ") for o in options]
        st.session_state["guess_labels"] = labels
        st.session_state["guess_index"] = random.randint(0, len(labels)-1)
        st.session_state["guess_correct_key"] = correct_key

    # 2. Display the *selected* image
    correct_key = st.session_state["guess_correct_key"]
    st.image(correct_key, width=250)

    # 3. Retrieve stable options
    options = st.session_state["guess_options"]
    labels = st.session_state["guess_labels"]

    # Stable radio (key ensures stability between reruns)
    index = st.session_state["guess_index"]
    selected_label = st.radio("Select the matching word or phrase:", labels, key="guess_radio", index=index)
    selected = next((o for o in options if o["label"].replace("_", " ") == selected_label), None)
    
    if st.button("Submit guess"):
        if selected["key"] == correct_key:
            st.success("Correct!")
            # reset for next slot
            st.session_state.pop("guess_options", None)
            st.session_state.pop("guess_correct_key", None)
            st.session_state.pop("guess_radio", None)
        else:
            st.error("Incorrect!")
            st.info(f"Correct answer: {Path(correct_key).stem}")

# Footer
# -----------------------
st.markdown("---")
st.caption("Per-side coords use bottom=23, left=15, top=22, right=14. Use capture tool to build CSV corrections quickly; upload CSV to merge and save corrected coordinates.")
