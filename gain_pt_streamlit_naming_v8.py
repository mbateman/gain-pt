# gain_pt_streamlit_naming_v8.py
"""
GAIN Naming Game v8 — multiplayer integration (manual selection)

Patched so that task submit feedback persists after rerun, tasks do not reappear
immediately, scores update reliably, and duplicate submit/button issues are removed.
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
import random

from synonyms import get_synonyms

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="GAIN PT Naming Game", layout="wide")
st.markdown(
    """
    <style>
    /* Entire app */
    html, body, [class*="css"]  {
        font-size: 28px !important;
    }

    /* Streamlit main app container */
    .stApp {
        font-size: 28px !important;
    }

    /* Inputs, buttons, labels */
    button, input, label, textarea, select {
        font-size: 18px !important;
    }

    /* Headers */
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.4rem !important; }

    /* Info / success / error boxes */
    div[data-testid="stAlert"] {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* Reduce vertical gap between stacked messages */
.stMarkdown p {
    margin-bottom: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)


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
    'sentence', 'guess', 'moveahead', 'sentence', 'naming', 'guess', 'naming',
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
    left = inner_margin
    right = img_w - inner_margin
    top = inner_margin
    bottom = img_h - inner_margin

    coords = []

    # Bottom side: right -> left
    if bottom_count > 1:
        xs = [right - i * (right - left) / (bottom_count - 1) for i in range(bottom_count)]
    else:
        xs = [(left + right) / 2.0]
    ys = [bottom] * bottom_count
    coords.extend(list(zip(xs, ys)))  # indices 0 .. bottom_count-1

    # Left side: bottom -> top
    if left_count > 1:
        ys_left = [bottom - i * (bottom - top) / (left_count - 1) for i in range(left_count)]
    else:
        ys_left = [(top + bottom) / 2.0]
    xs_left = [left] * left_count
    coords.extend(list(zip(xs_left, ys_left)))

    # Top side: left -> right
    if top_count > 1:
        xs_top = [left + i * (right - left) / (top_count - 1) for i in range(top_count)]
    else:
        xs_top = [(left + right) / 2.0]
    ys_top = [top] * top_count
    coords.extend(list(zip(xs_top, ys_top)))

    # Right side: top -> bottom
    if right_count > 1:
        ys_right = [top + i * (bottom - top) / (right_count - 1) for i in range(right_count)]
    else:
        ys_right = [(top + bottom) / 2.0]
    xs_right = [right] * right_count
    coords.extend(list(zip(xs_right, ys_right)))

    expected = bottom_count + left_count + top_count + right_count
    if expected != TOTAL_SLOTS:
        st.warning(f"Side counts sum {expected} != TOTAL_SLOTS {TOTAL_SLOTS}. Adjust counts.")
    coords = [(float(x), float(y)) for x,y in coords]
    return coords

# -----------------------
# Base coords loader/generator (per-side)
# -----------------------
def make_base_coords_by_side():
    board = load_board_image()
    W, H = board.size
    coords = compute_coords_by_side(W, H, bottom_count=23, left_count=15, top_count=22, right_count=14, inner_margin=60)
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

def is_correct_answer(user_text, target_word, slot):
    if not user_text:
        return False
    ui = user_text.strip().lower()
    exact_match = (target_word in ui) or (ui in target_word)
    if exact_match:
        return True
    else:
        synonyms = get_synonyms(ui) if slot == "naming" else get_synonyms(target_word)
        input = ui if slot == "naming" else target_word
        print(f"DEBUG: synonyms for {input}: {synonyms}")
        return input in synonyms

# -----------------------
# Session defaults (multiplayer)
# -----------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("players", ["Player 1", "Player 2"])
st.session_state.setdefault("max_players", 9)
# initialize per-player positions if missing
if "player_positions" not in st.session_state:
    st.session_state["player_positions"] = {p: 0 for p in st.session_state["players"]}
if "player_scores" not in st.session_state:
    st.session_state["player_scores"] = {p: 0 for p in st.session_state["players"]}
if "player_skip" not in st.session_state:
    st.session_state["player_skip"] = {p: False for p in st.session_state["players"]}
st.session_state.setdefault("current_player", st.session_state["players"][0])
st.session_state.setdefault("player_task_map", {})  # per-player task dicts
st.session_state.setdefault("token_colors", {})
st.session_state.setdefault("coord_overrides", {})
st.session_state.setdefault("captured_corrections", [])
st.session_state.setdefault("token_offset_x", DEFAULT_TOKEN_OFFSET_X)
st.session_state.setdefault("token_offset_y", DEFAULT_TOKEN_OFFSET_Y)
st.session_state.setdefault("last_feedback_message", "")
st.session_state.setdefault("last_feedback_type", "")  # "success", "error", "warning", "info"

# per-player last_task_position mapping
if "last_task_positions" not in st.session_state:
    st.session_state["last_task_positions"] = {p: None for p in st.session_state["players"]}

# default colours
_default_colors = ["#d9534f","#5cb85c","#5bc0de","#f0ad4e","#9440ed","#00b894","#ff6b6b","#ffe66d","#8e44ad"]
for i,p in enumerate(st.session_state["players"]):
    if p not in st.session_state["token_colors"]:
        st.session_state["token_colors"][p] = _default_colors[i % len(_default_colors)]

# -----------------------
# Load coords
# -----------------------
base_coords = load_base_coords()
corrected_coords = load_corrected_coords()
active_coords = corrected_coords  # used for drawing

# -----------------------
# Drawing & overlay helpers (updated to show all tokens)
# -----------------------
def draw_board_with_overlays(position_map=None, show_indices=False, show_overlay=False):
    """
    position_map: dict mapping player->position used to draw tokens; if None use current state
    """
    if position_map is None:
        position_map = st.session_state.get("player_positions", {})
    board = load_board_image().copy()
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    if show_overlay:
        for i,(x,y) in enumerate(base_coords):
            draw.text((x-14,y-14), str(i), fill="red", font=font)
        for i,(x,y) in enumerate(corrected_coords):
            draw.text((x+10,y+10), str(i), fill="green", font=font)
    elif show_indices:
        index_offset_x = 14
        index_offset_y = -14
        for i,(x,y) in enumerate(active_coords):
            draw.text((x - index_offset_x, y - index_offset_y), str(i), fill="black", font=font)

    # draw all tokens with slight spread
    players = list(st.session_state["players"])
    n = len(players)
    W, H = board.size
    for idx, player in enumerate(players):
        pos = int(st.session_state["player_positions"].get(player, 0))
        # guard
        if pos < 0 or pos >= len(active_coords):
            continue
        x, y = active_coords[pos]
        # apply global fine-tune offsets
        x = x + st.session_state.get("token_offset_x", 0)
        y = y + st.session_state.get("token_offset_y", 0)
        # spread tokens slightly around the slot deterministically
        spread = 12
        # Use index-based deterministic offsets so tokens don't jump each rerun
        dx = int(((idx % 3) - 1) * (spread))  # values -spread,0,+spread depending on idx mod 3
        dy = int(((idx // 3) - (n//6)) * 8)
        tx = x + dx
        ty = y + dy
        r = max(8, int(min(W, H) * 0.015))
        color = st.session_state["token_colors"].get(player, _default_colors[0])
        draw.ellipse((tx - r, ty - r, tx + r, ty + r), fill=color, outline="black")
        # initials
        initials = "".join([part[0].upper() for part in player.split()][:2])
        # write initials centered-ish
        draw.text((tx - (r/2), ty - (r/2)), initials, fill="white", font=font)

    return board

def draw_coordinate_grid(img, step=25, color=(0,0,0,80)):
    grid = img.copy()
    draw = ImageDraw.Draw(grid, "RGBA")
    W, H = grid.size
    for x in range(0, W, step):
        draw.line((x, 0, x, H), fill=color, width=1)
        draw.text((x+2, 2), str(x), fill=color)
    for y in range(0, H, step):
        draw.line((0, y, W, y), fill=color, width=1)
        draw.text((2, y+2), str(y))
    return grid

# -----------------------
# Calibration helper: per-selected-player current slot info & capture
# -----------------------
def get_selected_player():
    return st.session_state.get("current_player", st.session_state["players"][0])

def get_current_slot_info_for_player(player=None):
    if player is None:
        player = get_selected_player()
    idx = int(st.session_state["player_positions"].get(player, 0))
    slot_type = BASE_PATTERN[idx]
    x, y = active_coords[idx]
    return {"player": player, "index": idx, "slot_type": slot_type, "x": int(round(x)), "y": int(round(y))}

def capture_current_coordinate_for_selected():
    info = get_current_slot_info_for_player()
    cc = st.session_state.get("captured_corrections", [])
    token_offset_x = st.session_state.get("token_offset_x", 0)
    token_offset_y = st.session_state.get("token_offset_y", 0)
    # remove existing for this index for idempotency
    cc = [c for c in cc if c["index"] != info["index"]]
    cc.append({"index": info["index"], "x": info["x"] + token_offset_x, "y": info["y"] + token_offset_y})
    st.session_state["captured_corrections"] = sorted(cc, key=lambda r: r["index"])
    st.success(f"Captured index {info['index']} → ({info['x']+token_offset_x}, {info['y']+token_offset_y})")

def download_captured_csv():
    cc = st.session_state.get("captured_corrections", [])
    if not cc:
        st.warning("No captured corrections to download.")
        return None
    df = pd.DataFrame(cc)[["index","x","y"]]
    return df.to_csv(index=False)

# -----------------------
# Game movement & slot logic (per-player)
# -----------------------
def move_player(player, steps):
    pos = int(st.session_state["player_positions"].get(player, 0))
    new_pos = pos + int(steps)
    new_pos = max(0, min(TOTAL_SLOTS - 1, new_pos))
    # print(f"Moving {player} from {pos} to {new_pos} (steps: {steps})")
    st.session_state["player_positions"][player] = new_pos
    return new_pos

def apply_slot_effects_for_player(player):
    """
    Applies ALL slot effects:
    - moveahead → +3
    - moveback → -3
    - stop → skip next turn
    Updates player position immediately.
    """
    pos = int(st.session_state["player_positions"][player])
    slot = BASE_PATTERN[pos]

    if slot == "moveahead":
        # st.info(f"{player} landed on {slot.upper()}: +3 spaces")
        st.session_state["messages"].append(
            ("info", f"{player} landed on MOVE AHEAD: +3 spaces")
        )
        new_pos = move_player(player, 3)
        st.session_state["player_positions"][player] = new_pos

    elif slot == "moveback":
        # st.info(f"{player} landed on MOVE BACK: -3 spaces")
        st.session_state["messages"].append(
            ("info", f"{player} landed on MOVE BACK: -3 spaces")
        )
        new_pos = move_player(player, -3)
        st.session_state["player_positions"][player] = new_pos

    elif slot == "stop":
        st.info(f"{player} landed on STOP: will lose next turn")
        st.session_state["player_skip"][player] = True
        switch_user()

# -----------------------
# Helper: record feedback so it survives rerun
# -----------------------
def record_feedback(player, kind, message):
    # kind: "success"|"error"|"info"|"warning"
    st.session_state[f"last_feedback_{player}"] = {"kind": kind, "msg": message}

def show_and_clear_all_feedback():
    """
    Show feedback for ALL players that have pending messages.
    Only clear feedback AFTER it has been displayed.
    """
    to_clear = []
    for p in st.session_state["players"]:
        key = f"last_feedback_{p}"
        fb = st.session_state.get(key)
        if fb:
            k = fb.get("kind")
            m = fb.get("msg", "")
            if k == "success":
                st.success(f"{p}: {m}")
            elif k == "error":
                st.error(f"{p}: {m}")
            elif k == "warning":
                st.warning(f"{p}: {m}")
            else:
                st.info(f"{p}: {m}")
            to_clear.append(key)

    # Clear AFTER displaying.
    for key in to_clear:
        st.session_state.pop(key, None)


def switch_user():
    # -----------------------------------
    # TURN ROTATION AFTER TASK COMPLETION
    # -----------------------------------
    players = st.session_state["players"]
    current_index = players.index(current)

    # move to next player round-robin
    next_index = (current_index + 1) % len(players)
    next_player = players[next_index]

    # if next player was skipping a turn, consume the skip
    while st.session_state["player_skip"].get(next_player, False):
        st.session_state["player_skip"][next_player] = False  # skip consumed
        next_index = (next_index + 1) % len(players)
        next_player = players[next_index]

    st.session_state["current_player"] = next_player
    # -----------------------------------

# -----------------------
# Sidebar: calibration controls (kept from original)
# -----------------------
st.sidebar.header("Calibration & overlay")

show_indices = st.sidebar.checkbox("Show COORD indices (0–73)", value=False)
show_overlay = st.sidebar.checkbox("Show base vs corrected overlay", value=False)
show_coord_grid = st.sidebar.checkbox("Show coordinate grid", value=False)
show_fine_tune = st.sidebar.checkbox("Show fine tune", value=False)
show_quick_jump = st.sidebar.checkbox("Show quick jump to index", value=False)
roll_one = st.sidebar.checkbox("Roll dice one position forward", value=False)

uploaded = st.sidebar.file_uploader("Upload CSV corrections (index,x,y)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if {"index", "x", "y"}.issubset(df.columns):
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
# Main UI
# -----------------------
st.title("GAIN Naming Game — v8 (multiplayer)")

# col_controls, col_board = st.columns([1.2, 1.8], vertical_alignment="top")
col_controls, col_board = st.columns([1.25, 1.75], vertical_alignment="top")

with col_board:
    board_img = draw_board_with_overlays(position_map=st.session_state["player_positions"],
                                         show_indices=show_indices,
                                         show_overlay=show_overlay)
    if show_coord_grid:
        board_img = draw_coordinate_grid(board_img)
    # use width="stretch" following your original preference
    st.image(board_img, width="content")

with col_controls:
    st.markdown("### Players & Controls")
    st.markdown("**Players:**")
    for p in st.session_state["players"]:
        st.write(f"- {p} (pos: {st.session_state['player_positions'].get(p,0)}, score: {st.session_state['player_scores'].get(p,0)}, skip: {st.session_state['player_skip'].get(p,False)})")

    # Current player selection (manual)
    selected_ui_player = st.selectbox(
        "Current player (manual select):",
        st.session_state["players"],
        index=st.session_state["players"].index(st.session_state["current_player"])
    )
    # If user changed it manually → update state
    if selected_ui_player != st.session_state["current_player"]:
        st.session_state["current_player"] = selected_ui_player
        st.rerun()

    st.markdown("### Actions for selected player")
    current = st.session_state["current_player"]
    if st.session_state["player_skip"].get(current, False):
        st.info(f"{current} must skip this turn (clear skip to allow roll).")
        if st.button("Clear skip for selected player"):
            st.session_state["player_skip"][current] = False
            st.rerun()
    
    
    if st.button("🎲 Roll dice for selected player"):
        

        player_list = st.session_state["players"]
        current = st.session_state["current_player"]

        # CLEAR TASK MESSAGES
        st.session_state["messages"] = []

        # --- HANDLE SKIP ---
        if st.session_state["player_skip"].get(current, False):
            st.session_state["messages"].append(
                ("warning", f"{current} is skipping this turn.")
            )
            st.session_state["player_skip"][current] = False   # auto-clear skip
        else:
            # --- NORMAL ROLL ---
            r = 1 if roll_one else random.randint(1, 6)
            st.session_state[f"{current}_last_roll"] = r
            st.session_state["messages"].append(("info", f"{current} rolled a {r}"))

            move_player(current, r)
            apply_slot_effects_for_player(current)

            # Assign task if appropriate
            pos = st.session_state["player_positions"][current]
            slot_type = BASE_PATTERN[pos]
            if slot_type in {"naming", "sentence", "guess"}:
                if st.session_state["last_task_positions"].get(current) != pos:
                    img, word = random_task_image_and_word()
                    st.session_state["player_task_map"][current] = {
                        "pos": pos, "img": img, "word": word
                    }
                    st.session_state["last_task_positions"][current] = pos
            else:
                st.session_state["player_task_map"].pop(current, None)
                st.session_state["last_task_positions"][current] = None

        st.rerun()

    msg_area = st.empty()   # placeholder for messages
    # Render persistent messages
    with msg_area.container():
        for level, msg in st.session_state["messages"]:
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            elif level == "error":
                st.error(msg)
            elif level == "info":
                st.info(msg)

    # show_and_clear_all_feedback()
    
    # Fine tune / capture UI (per selected player)
    if show_fine_tune:
        st.markdown("### Token fine-tune")
        st.session_state["token_offset_x"] = st.slider("Offset X (px)", -120, 120, value=st.session_state.get("token_offset_x", 0), step=1)
        st.session_state["token_offset_y"] = st.slider("Offset Y (px)", -120, 120, value=st.session_state.get("token_offset_y", 0), step=1)

        st.markdown("### Calibration helper")
        curr = get_current_slot_info_for_player(current)
        st.write(f"Player: **{curr['player']}**")
        st.write(f"Index: **{curr['index']}**")
        st.write(f"Slot type: **{curr['slot_type']}**")
        st.write(f"Active coords: **{curr['x']}, {curr['y']}**")
        token_offset_x = st.session_state["token_offset_x"]
        token_offset_y = st.session_state["token_offset_y"]
        st.write(f"Adjusted coords: **{curr['x']+token_offset_x}, {curr['y']+token_offset_y}**")
        if st.button("📍 Capture coordinate for this index (selected)"):
            capture_current_coordinate_for_selected()
        captured = st.session_state.get("captured_corrections", [])
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
# Task pane (per selected player)
# -----------------------
current = get_selected_player()
pos = int(st.session_state["player_positions"].get(current, 0))

slot = BASE_PATTERN[pos]
st.markdown(f"**Selected player:** {current}")
st.markdown(f"**Position:** {pos} — Slot: {slot}")

# Show any feedback from previous submission (persistent across rerun)
show_and_clear_all_feedback()

if slot == "start":
    st.info("🎯 Start: on arrow — roll to begin.")
elif slot == "end":
    st.success("🏁 END reached — finished the circuit!")
    
elif slot == "naming":
    task = st.session_state["player_task_map"].get(current)
    if task: 
        if task.get("img"):
            st.image(task["img"], width=250)
        st.subheader("Naming")
        ans = st.text_input("What is this called?", key=f"naming_input_{current}")
        
        if st.button("Submit naming", key=f"submit_naming_{current}"):
            target = task["word"] if task else ""
            was_correct = is_correct_answer(ans, target, slot="naming")
            # update score safely (overwrite dict to trigger session_state change)
            scores = st.session_state["player_scores"]
            if was_correct:
                scores[current] = scores.get(current, 0) + 1
                record_feedback(current, "success", "✅ Correct")
            else:
                record_feedback(current, "error", f"❌ You entered {ans}. Expected: {target}")
            st.session_state["player_scores"] = scores

            # clear task and mark position as 'already served' so it doesn't reassigned
            st.session_state["player_task_map"].pop(current, None)
            st.session_state["last_task_positions"][current] = pos  # IMPORTANT: keep pos so no immediate reassign
            # clear any per-player input state
            st.session_state.pop(f"naming_input_{current}", None)
            switch_user()
            st.rerun()
    else:
        st.info(f"Roll the dice {st.session_state['current_player']}!")
        
elif slot == "sentence":
    task = st.session_state["player_task_map"].get(current)
    if task: 
        if task.get("img"):
            st.image(task["img"], width=250)
        st.subheader("Sentence")
        sent = st.text_area("Write a sentence using the target word", key=f"sentence_input_{current}")
        if st.button("Submit sentence", key=f"submit_sentence_{current}"):
            target = task["word"] if task else ""
            was_correct = is_correct_answer(sent, target, slot="sentence")
            scores = st.session_state["player_scores"]
            if was_correct:
                scores[current] = scores.get(current, 0) + 1
                record_feedback(current, "success", "✅ Contains target")
            else:
                record_feedback(current, "warning", f"⚠️ You entered {sent}. Try to include: {target}")
            st.session_state["player_scores"] = scores

            st.session_state["player_task_map"].pop(current, None)
            st.session_state["last_task_positions"][current] = pos  # IMPORTANT
            st.session_state.pop(f"sentence_input_{current}", None)
            switch_user()
            st.rerun()
    else:
        st.info(f"Roll the dice {st.session_state['current_player']}!")

elif slot == "guess":
    task = st.session_state["player_task_map"].get(current)
    
    # If task cleared already → hide UI
    if not task:
        st.info(f"Roll the dice {st.session_state['current_player']}!")
    else:
        st.subheader("Guess Task")
        # existing guess-state creation here

        # per-player guess state keys
        gopts_key = f"{current}_guess_options"
        glabels_key = f"{current}_guess_labels"
        gindex_key = f"{current}_guess_index"
        gcorrect_key = f"{current}_guess_correct_key"

        # initialise once per player's slot entry
        if gopts_key not in st.session_state:
            task = st.session_state["player_task_map"].get(current)
            correct_img = None
            if task:
                correct_img = task.get("img")
            # if no image assigned, try to assign one now
            if correct_img is None:
                img, word = random_task_image_and_word()
                st.session_state["player_task_map"][current] = {"pos": pos, "img": img, "word": word}
                correct_img = img

            correct_key = correct_img.as_posix() if correct_img else ""
            # choose distractors
            distractors = [p for p in SCENARIO_IMAGES if p != correct_img]
            distractors = random.sample(distractors, min(4, len(distractors)))
            options = []
            if correct_img:
                options.append({"label": correct_img.stem.replace("_"," "), "path": correct_img, "key": correct_key, "is_correct": True})
            for d in distractors:
                options.append({"label": d.stem.replace("_"," "), "path": d, "key": d.as_posix(), "is_correct": False})
            random.shuffle(options)
            st.session_state[gopts_key] = options
            labels = [o["label"] for o in options]
            st.session_state[glabels_key] = labels
            st.session_state[gindex_key] = 0
            st.session_state[gcorrect_key] = correct_key

        # display correct image (stable)
        correct_key = st.session_state.get(gcorrect_key, "")
        if correct_key:
            st.image(correct_key, width=250)

        options = st.session_state.get(gopts_key, [])
        labels = st.session_state.get(glabels_key, [])
        idx = st.session_state.get(gindex_key, 0)
        selected_label = st.radio("Select the matching word or phrase:", labels, key=f"guess_radio_{current}", index=idx)
        selected = next((o for o in options if o["label"] == selected_label), None)

        if st.button("Submit guess", key=f"submit_guess_{current}"):
            was_correct = False
            if selected and selected.get("key") == st.session_state.get(gcorrect_key):
                was_correct = True

            # update score safely
            scores = st.session_state["player_scores"]
            if was_correct:
                scores[current] = scores.get(current, 0) + 1
                record_feedback(current, "success", "Correct!")
            else:
                record_feedback(current, "error", "Incorrect!")
                try:
                    correct_label = Path(st.session_state.get(gcorrect_key, "")).stem
                except Exception:
                    correct_label = ""
                if correct_label:
                    # store info feedback; use info kind so it's not an error
                    record_feedback(current, "info", f"Correct answer: {correct_label}")

            st.session_state["player_scores"] = scores

            # ALWAYS clear per-player guess UI state and task
            for k in [gopts_key, glabels_key, gindex_key, gcorrect_key]:
                st.session_state.pop(k, None)
            # Also clear the radio widget key so Streamlit drops the old selection
            st.session_state.pop(f"guess_radio_{current}", None)

            st.session_state["player_task_map"].pop(current, None)
            # Mark this position as already served (so it won't be re-assigned until player moves)
            st.session_state["last_task_positions"][current] = pos
            
            switch_user()

            st.rerun()

# After task logic show any feedback stored during submission

if show_quick_jump:
    st.markdown("Quick jump to index (selected player):")
    j = st.number_input("Index (0-based):", min_value=0, max_value=TOTAL_SLOTS-1,
                        value=st.session_state["player_positions"].get(current, 0), key="jump_index")
    if st.button("Go to index for selected player"):
        st.session_state["player_positions"][current] = int(j)
        st.session_state["player_task_map"].pop(current, None)
        st.rerun()

# Status snapshot
st.markdown("Game status snapshot:")
for p in st.session_state["players"]:
    st.write(f"{p}: pos={st.session_state['player_positions'][p]}, score={st.session_state['player_scores'][p]}, skip={st.session_state['player_skip'][p]}")
