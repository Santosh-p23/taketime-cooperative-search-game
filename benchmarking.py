"""Benchmark learned vs original heuristic."""
import json, random
from take_time import GameState, create_deck, deal_cards, get_best_move, heuristic as original_heuristic, WIN_SCORE, LOSS_SCORE
from heuristic_learner import extract_features
import take_time as tt
import numpy as np

def load_learned_heuristic(path):
    with open(path) as f:
        data = json.load(f)
    w, m, s = data['weights'], data.get('scaler_mean'), data.get('scaler_std')
    def h(state):
        if state.is_terminal():
            return WIN_SCORE if state.check_win() else LOSS_SCORE
        f = extract_features(state)
        if m: f = (f - np.array(m)) / np.array(s)
        return sum(f[i] * w[k] for i, k in enumerate(w))
    return h

def simulate(heuristic_fn, depth=1):
    orig = tt.heuristic
    tt.heuristic = heuristic_fn
    try:
        deck = create_deck()
        hands, _ = deal_cards(deck)
        state = GameState(hands)
        while not state.is_terminal():
            move = get_best_move(state, depth)
            if move is None:
                moves = state.get_legal_moves()
                if not moves:
                    break
                move = random.choice(moves)
            state = state.apply_move(*move)
        return state.check_win()
    finally:
        tt.heuristic = orig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--weights", default="learned_wts_10000.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    learned_h = load_learned_heuristic(args.weights)
    lw = sum(simulate(learned_h, args.depth) for _ in range(args.games))
    ow = sum(simulate(original_heuristic, args.depth) for _ in range(args.games))
    
    print(f"Original: {ow}/{args.games} ({100*ow/args.games:.1f}%)")
    print(f"Learned: {lw}/{args.games} ({100*lw/args.games:.1f}%)")