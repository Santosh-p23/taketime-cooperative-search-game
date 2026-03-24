"""
heuristic_learner.py
====================
Learns optimal heuristic weights for Take Time by:
  1. Simulating N games (fast: depth=1 or random play)
  2. Recording feature vectors at every state + final outcome (win/loss)
  3. Training logistic regression on (features -> win probability)
  4. Extracting learned coefficients as new heuristic weights

Usage:
    python heuristic_learner.py                  # train with defaults
    python heuristic_learner.py --games 2000     # more training games
    python heuristic_learner.py --depth 2        # slightly stronger solver during data collection
    python heuristic_learner.py --no-bench       # skip benchmark (faster)
"""

import argparse
import random
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from take_time import (
    GameState, create_deck, deal_cards, get_best_move,
    heuristic as original_heuristic,
    TOTAL_CARDS, NUM_ZONES, ZONE1_COLOR, ZONE6_REQUIRED,
    WIN_SCORE, LOSS_SCORE,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTION
#     These are the same quantities the heuristic uses, returned as a numeric
#     vector instead of a weighted sum.  Each element corresponds to one
#     signal the heuristic cares about.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "ordering_reward_pairs",   # count of (i,j) pairs where sum_i <= sum_j
    "ordering_violation_gap",  # total excess sum_i - sum_j for violating pairs
    "empty_zone_shortfall",    # total shortfall from empty-zone ceiling risk
    "zone_coverage",           # number of non-empty zones
    "zone6_count",             # cards placed in zone 6
    "zone6_complete",          # 1 if zone 6 has >= ZONE6_REQUIRED cards
    "zone1_filled",            # 1 if zone 1 has at least one card
    "zone1_sum",               # sum of zone 1 (want low)
    "zone6_sum",               # sum of zone 6 (want high)
    "adjacent_gap_reward",     # total non-decreasing margin across adjacent zones
    "near_limit_penalty",      # total soft penalty for zones approaching 24
    "hard_violation",          # 1 if any zone exceeds 24
    "turns_remaining",         # cards left to play (normalised by TOTAL_CARDS)
]


def extract_features(state: GameState) -> np.ndarray:
    """Return a fixed-length feature vector for any game state."""
    sums     = state.zone_sums()
    counts   = state.zone_counts()
    rem_vals = state.all_remaining_values()
    best_rem = rem_vals[0] if rem_vals else 0

    # 1. Ordering
    ordering_reward = 0.0
    ordering_gap    = 0.0
    for i in range(NUM_ZONES):
        for j in range(i + 1, NUM_ZONES):
            si, sj = sums[i], sums[j]
            if si > 0 and sj > 0:
                if si <= sj:
                    ordering_reward += 1.0
                else:
                    ordering_gap += (si - sj)

    # 2. Empty-zone ceiling shortfall
    shortfall = 0.0
    for i in range(NUM_ZONES - 1):
        si = sums[i]
        if si == 0:
            continue
        for j in range(i + 1, NUM_ZONES):
            if sums[j] == 0:
                gap = si - best_rem
                if gap > 0:
                    shortfall += gap

    # 3. Coverage
    coverage = sum(1 for c in counts if c > 0)

    # 4. Zone 6
    z6_count    = counts[5]
    z6_complete = float(z6_count >= ZONE6_REQUIRED)

    # 5. Zone 1
    z1_filled = float(counts[0] > 0)
    z1_sum    = sums[0]

    # 6. Zone 6 sum
    z6_sum = sums[5]

    # 7. Adjacent gap reward
    adj_gap = sum(
        max(0.0, sums[i + 1] - sums[i])
        for i in range(NUM_ZONES - 1)
        if sums[i] > 0 and sums[i + 1] > 0
    )

    # 8. Near-limit soft penalty
    near_limit = sum(max(0, s - 20) for s in sums if s <= 24)

    # 9. Hard violation
    hard_viol = float(any(s > 24 for s in sums))

    # 10. Turns remaining (normalised)
    turns_rem = state.cards_remaining() / TOTAL_CARDS

    return np.array([
        ordering_reward,
        ordering_gap,
        shortfall,
        coverage,
        z6_count,
        z6_complete,
        z1_filled,
        z1_sum,
        z6_sum,
        adj_gap,
        near_limit,
        hard_viol,
        turns_rem,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA COLLECTION
#     depth=0 -> random play  (fastest, noisiest data)
#     depth=1 -> greedy/1-ply (fast, decent quality)
#     depth=2+ -> branch-and-bound (slow but strongest data)
# ══════════════════════════════════════════════════════════════════════════════

def _random_move(state: GameState):
    moves = state.get_legal_moves()
    return random.choice(moves) if moves else None


def simulate_one_game(depth: int = 1) -> tuple[list[np.ndarray], bool]:
    """Play one full game and return (list_of_feature_vectors, won)."""
    deck          = create_deck()
    hands, _      = deal_cards(deck)
    state         = GameState(hands)
    features_list = []

    while not state.is_terminal():
        features_list.append(extract_features(state))

        if depth == 0:
            move = _random_move(state)
        else:
            move = get_best_move(state, depth=depth)
            if move is None:
                move = _random_move(state)

        if move is None:
            break
        state = state.apply_move(*move)

    won = state.check_win()
    return features_list, won


def collect_data(n_games: int = 1000, depth: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_games simulations.
    Returns X (n_samples, n_features) and y (n_samples,) as numpy arrays.
    No fixed seed — fresh randomness every run.
    """
    X_list, y_list = [], []
    wins = 0

    print(f"Collecting data from {n_games} games (depth={depth})...")
    for i in range(n_games):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_games}  wins so far: {wins}")

        features_list, won = simulate_one_game(depth=depth)
        label = 1 if won else 0
        if won:
            wins += 1

        for fv in features_list:
            X_list.append(fv)
            y_list.append(label)

    print(f"\nDone.  Win rate during collection: {wins}/{n_games} ({100*wins/n_games:.1f}%)")
    print(f"Total state samples: {len(X_list)}")
    return np.array(X_list), np.array(y_list)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN + EXTRACT WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def train_and_extract_weights(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train logistic regression on game-state features -> win/loss.
    Returns a dict with the learned weights and training metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["loss", "win"])

    print(f"\n── Model Accuracy: {acc:.3f} ──")
    print(report)

    # Un-scale coefficients so they work in original feature units
    raw_coefs      = clf.coef_[0]
    unscaled_coefs = raw_coefs / scaler.scale_

    weights = {name: float(w) for name, w in zip(FEATURE_NAMES, unscaled_coefs)}

    print("\n── Learned Weights (unscaled, plug into heuristic) ──")
    for name, w in weights.items():
        direction = "↑ good" if w > 0 else "↓ bad "
        print(f"  {direction}  {name:<30}  {w:+.4f}")

    return {
        "weights":     weights,
        "accuracy":    acc,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std":  scaler.scale_.tolist(),
        "coef_scaled": raw_coefs.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LEARNED HEURISTIC
# ══════════════════════════════════════════════════════════════════════════════

def make_learned_heuristic(weights: dict):
    """
    Returns a heuristic function using the learned weights.
    Drop-in replacement for take_time.heuristic().
    """
    w = weights

    def learned_heuristic(state: GameState) -> float:
        if state.is_terminal():
            return WIN_SCORE if state.check_win() else LOSS_SCORE

        sums = state.zone_sums()
        if any(s > 24 for s in sums):
            return LOSS_SCORE

        # Zone 1 color is a hard constraint — keep it explicit
        counts = state.zone_counts()
        if counts[0] > 0:
            if not all(c[1] == ZONE1_COLOR for c in state.zones[1]):
                return LOSS_SCORE

        fv    = extract_features(state)
        score = sum(fv[i] * w[FEATURE_NAMES[i]] for i in range(len(FEATURE_NAMES)))

        # Scale up to match original score magnitudes
        return score * 100.0

    return learned_heuristic


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(weights: dict, n_eval: int = 200, depth: int = 1) -> None:
    """Compare win rates: original heuristic vs learned heuristic."""
    import take_time as tt

    # Capture original BEFORE any patching to avoid infinite recursion
    learned_h = make_learned_heuristic(weights)

    def run_games(heuristic_fn, label: str) -> float:
        tt.heuristic = heuristic_fn
        wins = 0
        for _ in range(n_eval):
            deck = create_deck()
            hands, _ = deal_cards(deck)
            state = GameState(hands)
            while not state.is_terminal():
                move = get_best_move(state, depth=depth)
                if move is None:
                    move = _random_move(state)
                if move is None:
                    break
                state = state.apply_move(*move)
            if state.check_win():
                wins += 1
        tt.heuristic = original_heuristic  # restore after each run
        wr = wins / n_eval
        print(f"  {label:<25}  win rate: {wins}/{n_eval} ({100*wr:.1f}%)")
        return wr

    print(f"\n── Benchmark ({n_eval} games, depth={depth}) ──")
    orig_wr    = run_games(original_heuristic, "Original heuristic")
    learned_wr = run_games(learned_h,          "Learned heuristic")
    delta      = (learned_wr - orig_wr) * 100
    print(f"\n  Delta: {delta:+.1f}pp  {'(improvement!)' if delta > 0 else '(original is better)'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Learn heuristic weights for Take Time")
    parser.add_argument("--games",    type=int, default=1000,                   help="Number of training games (default: 1000)")
    parser.add_argument("--depth",    type=int, default=1,                      help="Solver depth during data collection (0=random, 1=greedy, 2+=BnB)")
    parser.add_argument("--eval",     type=int, default=200,                    help="Games for benchmark evaluation")
    parser.add_argument("--out",      type=str, default="learned_weights.json", help="Output file for weights")
    parser.add_argument("--no-bench", action="store_true",                      help="Skip benchmark (faster)")
    args = parser.parse_args()

    X, y   = collect_data(n_games=args.games, depth=args.depth)
    result = train_and_extract_weights(X, y)

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWeights saved to {args.out}")

    if not args.no_bench:
        benchmark(result["weights"], n_eval=args.eval, depth=args.depth)


if __name__ == "__main__":
    main()