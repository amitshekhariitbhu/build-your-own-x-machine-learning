import numpy as np

# Bar Chart Race Visualization, from scratch (no plotting library).
# A bar chart race animates a ranked set of bars whose values change over
# time: categories rise, fall, and OVERTAKE one another, and the bars are
# re-sorted every frame. We build it from three pieces of plain math --
#   (1) linear INTERPOLATION between yearly keyframes for smooth motion,
#   (2) a per-frame argsort RANKING (the "racing" part),
#   (3) proportional ASCII BAR rendering scaled to the current leader.
# Each frame is just text. Correctness is proven by exact, hand-verifiable
# checks that clearly beat trivial baselines (fixed-order / random / step).

CATEGORIES = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot']
BAR_WIDTH = 40                     # characters for the longest bar in a frame


def make_keyframes():
    """Synthetic yearly totals with a PLANTED full-reversal narrative.

    Each brand moves linearly from its own start value to its own final
    value. 'Alpha' leads at the start but declines; 'Foxtrot' starts dead
    last but grows fastest and finishes #1 -- a known ground-truth ranking
    trajectory (with lots of overtakes in between) for the race to recover."""
    np.random.seed(0)
    years = np.arange(2000, 2011)                        # 11 keyframe years
    t = (years - years[0]).astype(float)
    starts = np.array([95., 80., 68., 55., 40.,  8.])    # Alpha top, Foxtrot last
    finals = np.array([70., 82., 95., 110., 125., 140.]) # fully reversed order
    rates = (finals - starts) / t[-1]
    values = starts[None, :] + rates[None, :] * t[:, None]
    values += np.random.uniform(-1.5, 1.5, size=values.shape)   # small wiggle
    return years, values


def interpolate(values, fps):
    """Linear interpolation between consecutive keyframes -> smooth frames.
    Returns frames (F, N) and the fractional keyframe-time of each frame."""
    K = values.shape[0]
    frames, ftime = [], []
    for k in range(K - 1):
        for f in range(fps):
            a = f / fps
            frames.append((1 - a) * values[k] + a * values[k + 1])
            ftime.append(k + a)
    frames.append(values[-1].copy())                     # final keyframe
    ftime.append(float(K - 1))
    return np.array(frames), np.array(ftime)


def render_frame(vals, title):
    """Sort bars descending and draw them proportional to the leader."""
    order = np.argsort(vals)[::-1]
    leader = vals[order[0]]
    lines = [title]
    for i in order:
        length = int(round(BAR_WIDTH * vals[i] / leader))
        lines.append(f"{CATEGORIES[i]:>8} | {'#' * length} {vals[i]:6.1f}")
    return lines


def decode_frame(lines):
    """Parse a rendered frame back into (order, bar_lengths) from pure text --
    proves the ASCII picture faithfully encodes rank and magnitude."""
    order, lengths = [], []
    for line in lines[1:]:
        label, bar = line.split('|')
        order.append(label.strip())
        lengths.append(bar.count('#'))
    return order, np.array(lengths)


if __name__ == "__main__":
    years, keyvals = make_keyframes()
    fps = 5
    frames, ftime = interpolate(keyvals, fps)
    names = np.array(CATEGORIES)
    n, F = len(CATEGORIES), len(frames)

    # ---- Show the race at start / middle / end ----
    for idx in (0, F // 2, F - 1):
        for ln in render_frame(frames[idx], f"\n=== year {years[0] + ftime[idx]:5.1f} ==="):
            print(ln)

    # ---- Check 1: ranking recovered from the ASCII vs baselines ----
    # racing renderer re-sorts every frame; a NON-racing bar chart keeps the
    # original fixed order; random guessing gets 1/n per slot.
    rank_ok = fixed_ok = 0
    for fr in frames:
        shown, _ = decode_frame(render_frame(fr, "t"))
        truth = list(names[np.argsort(fr)[::-1]])
        rank_ok += sum(a == b for a, b in zip(shown, truth))
        fixed_ok += sum(a == b for a, b in zip(list(names), truth))
    rank_acc, fixed_acc, rand_acc = rank_ok / (F * n), fixed_ok / (F * n), 1.0 / n

    # ---- Check 2: bars are proportional (round-trip decode of magnitude) ----
    max_prop_err = 0.0
    for fr in frames:
        _, lengths = decode_frame(render_frame(fr, "t"))
        order = np.argsort(fr)[::-1]
        want = fr[order] / fr[order][0]                  # true normalized magnitude
        max_prop_err = max(max_prop_err, np.max(np.abs(lengths / BAR_WIDTH - want)))

    # ---- Check 3: interpolation exactness vs a hold-last (step) baseline ----
    key_idx = np.arange(keyvals.shape[0]) * fps          # frames on keyframes
    interp_err = np.max(np.abs(frames[key_idx] - keyvals))
    step_pred = keyvals[np.clip(ftime.astype(int), 0, keyvals.shape[0] - 1)]
    step_err = np.mean(np.abs(step_pred - frames))       # baseline misses motion

    # ---- Check 4: the planted overtake narrative ----
    start_leader, end_leader = names[np.argmax(frames[0])], names[np.argmax(frames[-1])]
    end_order = list(names[np.argsort(frames[-1])[::-1]])
    planted_end = ['Foxtrot', 'Echo', 'Delta', 'Charlie', 'Bravo', 'Alpha']

    print("\n--- correctness signal (bar chart race) ---")
    print(f"frames rendered ............. {F}  ({n} categories, {years[0]}-{years[-1]})")
    print(f"rank accuracy (racing) ...... {rank_acc:.3f}   <- decoded from ASCII")
    print(f"rank accuracy (fixed-order) . {fixed_acc:.3f}   (no re-sort baseline)")
    print(f"rank accuracy (random) ...... {rand_acc:.3f}   (1/n chance level)")
    print(f"bar proportional error ...... {max_prop_err:.4f}  (<= {0.5/BAR_WIDTH:.4f} rounding)")
    print(f"interp error at keyframes ... {interp_err:.2e}  vs step baseline {step_err:.2f}")
    print(f"leader: {start_leader} (start) -> {end_leader} (end)   full reversal recovered: {end_order == planted_end}")

    ok = (rank_acc == 1.0 and rank_acc > fixed_acc > rand_acc - 1e-9
          and max_prop_err <= 0.5 / BAR_WIDTH + 1e-9 and interp_err < 1e-9
          and step_err > 1.0 and start_leader == 'Alpha'
          and end_leader == 'Foxtrot' and end_order == planted_end)
    print("RESULT:", "PASS -- race renders faithfully and beats baselines" if ok else "FAIL")
