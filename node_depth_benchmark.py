#!/usr/bin/env python3
"""
Average nodes explored per game across solver depths.

Uses the same move logic as take_time.get_best_move but calls cooperative_search
directly so node counts can be summed without modifying take_time.py.

Depth 0: random legal move (0 search nodes).
Depth >= 1: branch-and-bound as in take_time.
"""

from __future__ import annotations

import argparse
import random
import statistics
from typing import List, Optional, Tuple

from take_time import (
    GameState,
    Move,
    TOTAL_CARDS,
    cooperative_search,
    create_deck,
    deal_cards,
    heuristic,
)


def choose_move_with_nodes(
    state: GameState, depth: int
) -> Tuple[Optional[Move], int]:
    """
    Mirror take_time.get_best_move; return (move, nodes_this_call).
    """
    if depth == 0:
        moves = state.get_legal_moves()
        if not moves:
            return None, 0
        return random.choice(moves), 0

    nodes = [0]
    _score, seq = cooperative_search(state, depth, float("-inf"), nodes)

    if seq:
        _, card, zone = seq[0]
        return (card, zone), nodes[0]

    moves = state.get_legal_moves()
    if not moves:
        return None, nodes[0]
    mv = max(moves, key=lambda m: heuristic(state.apply_move(*m)))
    return mv, nodes[0]


def play_one_game(depth: int) -> int:
    """Return total cooperative_search nodes expended in one full game."""
    deck = create_deck()
    hands, _ = deal_cards(deck)
    state = GameState(hands)
    total = 0
    while not state.is_terminal():
        move, n = choose_move_with_nodes(state, depth)
        total += n
        if move is None:
            moves = state.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
        state = state.apply_move(*move)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average search nodes per game vs solver depth"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games to average per depth (default: 50)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated depths to benchmark (default: 0,1,2,3,4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    depths: List[int] = []
    for part in args.depths.split(","):
        part = part.strip()
        if not part:
            continue
        depths.append(int(part))

    if not depths:
        raise SystemExit("No depths given")

    random.seed(args.seed)

    print(f"Games per depth: {args.games}  |  seed: {args.seed}")
    print(f"Turns per game: {TOTAL_CARDS}")
    print()
    print(f"{'depth':>6}  {'avg_nodes/game':>18}  {'stdev':>12}")
    print("-" * 42)

    for d in depths:
        per_game = [play_one_game(d) for _ in range(args.games)]
        avg = statistics.mean(per_game)
        dev = statistics.stdev(per_game) if args.games > 1 else 0.0
        print(f"{d:>6}  {avg:18,.1f}  {dev:12,.1f}")


if __name__ == "__main__":
    main()
