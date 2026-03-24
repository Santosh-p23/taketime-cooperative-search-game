import json
import sys
from take_time import GameState, get_best_move, upper_bound, LOSS_SCORE, _fc, heuristic, TOTAL_CARDS

def load_state(filepath: str) -> GameState:
    with open(filepath, 'r') as f:
        data = json.load(f)

    hands = [
        [(v, c) for v, c in hand]
        for hand in data['hands']
    ]

    state = GameState(hands)
    state.zones = {
        int(z): [(v, c) for v, c in cards]
        for z, cards in data['zones'].items()
    }
    state.turn = data['turn']
    return state

def next_best_move(state: GameState) -> dict:
    """Return the single best move for the current player."""

    if not state.get_legal_moves():
        return {"verdict": "LOSS", "reason": "No legal moves available"}

    if upper_bound(state) <= LOSS_SCORE:
        return {"verdict": "LOSS", "reason": "Game state is unwinnable"}

    move = get_best_move(state, depth=4)

    if move is None:
        return {"verdict": "LOSS", "reason": "Solver could not find a move"}

    card, zone = move
    return {
        "verdict": "OK",
        "card": {"value": card[0], "color": card[1]},
        "zone": zone
    }


def simulate_from_state(state: GameState, depth: int = 4) -> dict:
    """
    Simulate the rest of the game from a given state.
    Returns every move made and the final verdict.
    """
    moves_made = []

    while not state.is_terminal():
        if not state.get_legal_moves():
            return {
                "verdict": "LOSS",
                "reason": "No legal moves available",
                "moves_made": moves_made
            }

        if upper_bound(state) <= LOSS_SCORE:
            return {
                "verdict": "LOSS",
                "reason": "Game state is unwinnable",
                "moves_made": moves_made
            }

        move = get_best_move(state, depth=depth)
        if move is None:
            return {
                "verdict": "LOSS",
                "reason": "Solver could not find a move",
                "moves_made": moves_made
            }

        card, zone = move
        moves_made.append({
            "player": state.current_player + 1,
            "card": {"value": card[0], "color": card[1]},
            "zone": zone
        })
        state = state.apply_move(card, zone)

    # Game over — check result
    sums   = state.zone_sums()
    counts = state.zone_counts()

    return {
        "verdict": "WIN" if state.check_win() else "LOSS",
        "moves_made": moves_made,
        "final_board": {
            "sums":   sums,
            "counts": counts,
        }
    }


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "gamestate_example.json"
    state = load_state(filepath)

    print("=== NEXT BEST MOVE ===")
    print(json.dumps(next_best_move(state), indent=2))

    print("\n=== SIMULATE FROM CURRENT STATE ===")
    print(json.dumps(simulate_from_state(state, depth=4), indent=2))