"""
  Algorithm: Cooperative Minimax Branch-and-Bound                        
   • All 3 players share one maximizing objective (no MIN node) 
   • Pruning: upper_bound(state) ≤ alpha  → cut branch          
   • Move ordering (best-first) fires cuts as early as possible 
"""

import random
from typing import List, Tuple, Optional

Color = str
Card  = Tuple[int, Color]
Move  = Tuple[Card, int]

NUM_ZONES        = 6
NUM_PLAYERS      = 3
CARDS_PER_PLAYER = 4
TOTAL_CARDS      = NUM_PLAYERS * CARDS_PER_PLAYER
ZONE1_COLOR      = 'W'
ZONE6_REQUIRED   = 3
WIN_SCORE        = 10_000.0
LOSS_SCORE       = -10_000.0



# 1.  DECK & DEAL

def create_deck() -> List[Card]:
    return [(w, 'W') for w in range(1, 13)] + [(b, 'B') for b in range(1, 13)]


def deal_cards(deck: List[Card]) -> Tuple[List[List[Card]], List[Card]]:
    shuffled = deck[:]
    random.shuffle(shuffled)
    dealt  = shuffled[:TOTAL_CARDS]
    unused = shuffled[TOTAL_CARDS:]
    hands  = [dealt[i * CARDS_PER_PLAYER:(i + 1) * CARDS_PER_PLAYER]
              for i in range(NUM_PLAYERS)]
    return hands, unused



# 2.  GAME STATE

class GameState:
    """
    Full (god's-eye) game state for the cooperative solver.

    In real play, each player can only see their own card values and
    the colours of their teammates' cards.  The solver operates over
    the full state to compute the globally optimal cooperative plan.
    """

    def __init__(self, hands: List[List[Card]]):
        self.hands        = [list(h) for h in hands]
        self.zones        = {z: [] for z in range(1, NUM_ZONES + 1)}
        self.turn         = 0
        self.move_history = []

    @property
    def current_player(self) -> int:
        return self.turn % NUM_PLAYERS

    def is_terminal(self) -> bool:
        return self.turn == TOTAL_CARDS

    def cards_remaining(self) -> int:
        return TOTAL_CARDS - self.turn

    def zone_sums(self) -> List[int]:
        return [sum(c[0] for c in self.zones[z]) for z in range(1, NUM_ZONES + 1)]

    def zone_counts(self) -> List[int]:
        return [len(self.zones[z]) for z in range(1, NUM_ZONES + 1)]

    def all_remaining_values(self) -> List[int]:
        """All card values still held by any player, sorted descending."""
        vals = sorted((v for hand in self.hands for v, _ in hand), reverse=True)
        return vals

    # ── Feasibility ───────────────────────────────────────────────
    
    #Lookahead check for zone6, if cards required for zone6 can be fulfilled by remaining cards
    def _zone6_feasible(self, target: int) -> bool:
        z6_after  = len(self.zones[6]) + (1 if target == 6 else 0)
        needed = max(0, ZONE6_REQUIRED - z6_after)
        remaining = self.cards_remaining() - 1
        return needed <= remaining

    def _non_zone6_fillable(self, target: int) -> bool:
        unfilled = sum(
            1 for z in range(1, 6)
            if len(self.zones[z]) == 0 and z != target
        )
        remaining  = self.cards_remaining() - 1
        z6_after   = len(self.zones[6]) + (1 if target == 6 else 0)
        z6_needed  = max(0, ZONE6_REQUIRED - z6_after)
        return (remaining - z6_needed) >= unfilled

    # ── Legal moves ───────────────────────────────────────────────

    def is_legal_move(self, card: Card, zone: int) -> bool:
        _, color = card
        if card not in self.hands[self.current_player]:         return False
        if zone == 1 and color != ZONE1_COLOR:                  return False
        if not self._zone6_feasible(zone):                      return False
        if not self._non_zone6_fillable(zone):                  return False
        return True

    def get_legal_moves(self) -> List[Move]:
        hand  = self.hands[self.current_player]
        moves, seen = [], set()
        for card in hand:
            for zone in range(1, NUM_ZONES + 1):
                key = (card, zone)
                if key not in seen and self.is_legal_move(card, zone):
                    moves.append((card, zone))
                    seen.add(key)
        return moves

    # ── Transition ────────────────────────────────────────────────

    def apply_move(self, card: Card, zone: int) -> 'GameState':
        ns              = GameState.__new__(GameState)
        ns.hands        = [list(h) for h in self.hands]
        ns.zones        = {z: list(cs) for z, cs in self.zones.items()}
        ns.turn         = self.turn + 1
        ns.move_history = self.move_history + [(self.current_player, card, zone)]
        ns.hands[self.current_player].remove(card)
        ns.zones[zone].append(card)
        return ns

    # ── Win condition ─────────────────────────────────────────────

    def check_win(self) -> bool:
        if not self.is_terminal():
            return False
        counts = self.zone_counts()
        if any(c == 0 for c in counts):                          return False
        if counts[5] < ZONE6_REQUIRED:                          return False
        if any(c[1] != ZONE1_COLOR for c in self.zones[1]):     return False
        sums = self.zone_sums()
        if any(s > 24 for s in sums):
            return False
        
        return all(sums[i] <= sums[i + 1] for i in range(NUM_ZONES - 1))



# 3.  HEURISTIC Strategies

def heuristic(state: GameState) -> float:
    """
    heuristic function to evaluate the winning chance for each state
    currently uses static weight values to reward or penalize the score for different conditions
    """
    
    if state.is_terminal():
        return WIN_SCORE if state.check_win() else LOSS_SCORE

    sums     = state.zone_sums()
    counts   = state.zone_counts()
    score    = 0.0
    rem_vals = state.all_remaining_values()
    best_rem = rem_vals[0] if rem_vals else 0

    # 1. All-pairs ordering check
    for i in range(NUM_ZONES):
        for j in range(i + 1, NUM_ZONES):
            s_i, s_j = sums[i], sums[j]
            if s_i > 0 and s_j > 0:
                if s_i <= s_j:
                    score += 25.0
                else:
                    score -= 200.0 + 5.0 * (s_i - s_j)

    # 2. Empty-zone ceiling risk
    #    Filled zone Z has sum S; later zone J is empty — J must eventually
    #    reach ≥ S.  If best remaining card < S, penalise the gap.
    for i in range(NUM_ZONES - 1):
        s_i = sums[i]
        if s_i == 0:
            continue
        for j in range(i + 1, NUM_ZONES):
            if sums[j] == 0:
                shortfall = s_i - best_rem
                if shortfall > 0:
                    score -= shortfall * 6.0

    # 3. Zone coverage
    score += sum(1 for c in counts if c > 0) * 10.0

    # 4. Zone 6 progress
    score += counts[5] * 12.0
    if counts[5] >= ZONE6_REQUIRED:
        score += 40.0

    # 5. Zone 1 colour correctness, already checked in legal moves, this is for sanity check if heuristic is called without legal moves
    if counts[0] > 0:
        score += 15.0 if all(c[1] == ZONE1_COLOR for c in state.zones[1]) else LOSS_SCORE

    # 6. Zone 1 sum: keep small to set a low floor for the whole chain
    if sums[0] > 0:
        score -= sums[0] * 2.0

    # 7. Zone 6 sum: reward large values (sets the ceiling)
    if sums[5] > 0:
        score += sums[5] * 0.5

    # 8. Adjacent gap reward (non-decreasing margin)
    for i in range(NUM_ZONES - 1):
        if sums[i] > 0 and sums[i + 1] > 0:
            score += max(0.0, sums[i + 1] - sums[i]) * 0.5
    
    # 9. Penalize zones approaching or exceeding 24
    for s in sums:
        if s > 24:
            score -= 2000.0   # hard violation
        else:
            score -= max(0, s - 20) * 5.0   # soft penalty near limit

    return score



# 4.  UPPER BOUND ESTIMATOR

def upper_bound(state: GameState) -> float:
    """
    Optimistic upper bound — returns LOSS_SCORE if any hard constraint
    is already irrecoverable, even with perfect remaining play.

    Pruning rule:  upper_bound(state) ≤ alpha  → prune branch.
    """
    if state.is_terminal():
        return heuristic(state)

    sums      = state.zone_sums()
    counts    = state.zone_counts()
    remaining = state.cards_remaining()
    
    # (a) Zone sum exceeds limit → impossible to fix
    if any(s > 24 for s in sums):
        return LOSS_SCORE

    # (b) Zone 6 card count unachievable
    z6_needed = max(0, ZONE6_REQUIRED - counts[5])
    if z6_needed > remaining:
        return LOSS_SCORE

    # (c) Insufficient turns to fill all zones 1-5
    unfilled_non6 = sum(1 for z in range(1, 6) if counts[z - 1] == 0)
    if (remaining - z6_needed) < unfilled_non6:
        return LOSS_SCORE

    # (d) Cross-zone violation where even throwing ALL remaining cards
    #     at the smaller zone can't close the gap
    rem_vals  = state.all_remaining_values()
    total_rem = sum(rem_vals)
    for i in range(NUM_ZONES):
        for j in range(i + 1, NUM_ZONES):
            s_i, s_j = sums[i], sums[j]
            if 0 < s_j < s_i:
                if s_j + total_rem < s_i:
                    return LOSS_SCORE

    return heuristic(state) + remaining * 40.0



# 5.  MOVE ORDERING
def order_moves(state: GameState, moves: List[Move]) -> List[Move]:
    """
    Sort moves by immediate heuristic gain (best first).

    Best-first ordering is the same mechanism that makes standard
    alpha-beta efficient: the search sees good moves early, updates
    alpha quickly, and prunes more branches thereafter.
    """
    return sorted(
        moves,
        key=lambda mv: heuristic(state.apply_move(*mv)),
        reverse=True,
    )



# 6.  COOPERATIVE SEARCH
def cooperative_search(
    state:         GameState,
    depth:         int,
    alpha:         float,
    nodes_visited: List[int],
) -> Tuple[float, List[Tuple[int, Card, int]]]:
    """
    Cooperative branch-and-bound with alpha-beta–style pruning.

    There is no MIN node.  All players maximise together.
    Pruning fires when the optimistic upper bound of a branch can't
    improve on alpha (the best score already guaranteed elsewhere).

    Returns (best_score, move_sequence)
    where move_sequence = [(player, card, zone), …]
    """
    nodes_visited[0] += 1

    if state.is_terminal() or depth == 0:
        return heuristic(state), []

    ub = upper_bound(state)
    if ub <= alpha:           # This branch can't beat what we already have
        return ub, []

    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return heuristic(state), []

    ordered       = order_moves(state, legal_moves)
    best_value    = float('-inf')
    best_sequence = []

    for card, zone in ordered:
        next_state     = state.apply_move(card, zone)
        value, sub_seq = cooperative_search(next_state, depth - 1, alpha, nodes_visited)

        if value > best_value:
            best_value    = value
            best_sequence = [(state.current_player, card, zone)] + sub_seq

        if best_value > alpha:
            alpha = best_value

        if best_value >= WIN_SCORE:   # Confirmed win — stop early
            break

    return best_value, best_sequence




def get_best_move(state: GameState, depth: int = 5) -> Optional[Move]:
    """
    Return the best (card, zone) for the current player.

    Parameters
    ----------
    state : Full game state (god's-eye view for the solver).
    depth : Search depth.  5 is a solid default; 7+ for stronger play.
    """
    nodes = [0]
    score, seq = cooperative_search(state, depth, float('-inf'), nodes)
    print(f"    ↳ nodes: {nodes[0]:,}   best score: {score:.1f}")

    if seq:
        _, card, zone = seq[0]
        return card, zone

    # Greedy fallback
    moves = state.get_legal_moves()
    return max(moves, key=lambda mv: heuristic(state.apply_move(*mv))) if moves else None


# 7. SIMULATION & REPORTING

def _fc(card: Card) -> str:
    v, c = card
    return f"{v}{'W' if c == 'W' else 'B'}"


def _print_board(state: GameState) -> None:
    sums   = state.zone_sums()
    counts = state.zone_counts()
    sep    = "─" * 66
    print(f"\n  {sep}")
    print(f"  Turn {state.turn:>2}/12  │  Player {state.current_player + 1} to move")
    print(f"  {sep}")
    print("  Zone     │ " + " │ ".join(f" Z{z} " for z in range(1, NUM_ZONES + 1)))
    print("  Cards    │ " + " │ ".join(f"  {c} " for c in counts))
    print("  Sum      │ " + " │ ".join(f" {s:2} " for s in sums))
    print(f"  {sep}")
    for p in range(NUM_PLAYERS):
        hand_str = "  ".join(_fc(c) for c in state.hands[p])
        marker   = " ◀ to play" if p == state.current_player else ""
        print(f"  Player {p + 1}: [{hand_str}]{marker}")
    print(f"  {sep}")


def simulate_game(depth: int = 5, seed: Optional[int] = None) -> GameState:
    """Run a complete cooperative game and print the full transcript."""
    if seed is not None:
        random.seed(seed)

    deck          = create_deck()
    hands, unused = deal_cards(deck)

    print("\n" + "═" * 66)
    print("  COOPERATIVE SEARCH GAME - Take Time")
    print("═" * 66)
    print("\n  [DEAL]")
    for i, hand in enumerate(hands):
        print(f"  Player {i + 1}: {' '.join(_fc(c) for c in hand)}")
        print(f"           → teammates see colors: [{' '.join(c for _, c in hand)}]")
    print(f"  Unused (hidden): {len(unused)} cards\n")

    state = GameState(hands)
    print(f"  [PLACEMENT  depth={depth}]")

    for _ in range(TOTAL_CARDS):
        _print_board(state)
        player = state.current_player
        print(f"\n  Player {player + 1} thinking…")

        move = get_best_move(state, depth=depth)
        if move is None:
            print("  !! No legal moves — game stuck!")
            break

        card, zone = move
        print(f"  ✓  Place {_fc(card)} → Zone {zone}")
        state = state.apply_move(card, zone)

    # ── Result ──
    print("\n" + "═" * 66)
    print("  FINAL BOARD")
    print("═" * 66)
    sums   = state.zone_sums()
    counts = state.zone_counts()
    for z in range(1, NUM_ZONES + 1):
        cards_str = "  ".join(_fc(c) for c in state.zones[z])
        print(f"  Zone {z} ({counts[z-1]} card{'s' if counts[z-1]!=1 else ''}):"
              f"  [{cards_str}]  → sum = {sums[z-1]}")

    is_sorted  = all(sums[i] <= sums[i + 1] for i in range(NUM_ZONES - 1))
    z6_ok      = (counts[5] >= ZONE6_REQUIRED)
    all_filled = all(c > 0 for c in counts)
    z1_ok      = bool(state.zones[1]) and all(c[1] == ZONE1_COLOR for c in state.zones[1])

    print(f"\n  Sums              : {sums}")
    print(f"  Non-decreasing    : {'✓' if is_sorted  else '✗'}")
    print(f"  Zone 6 = 3 cards  : {'✓' if z6_ok      else '✗'}")
    print(f"  All zones filled  : {'✓' if all_filled  else '✗'}")
    print(f"  Zone 1 all-white  : {'✓' if z1_ok       else '✗'}")
    result = "🏆  WIN" if state.check_win() else "✗  LOSS"
    print(f"\n  RESULT: {result}")
    print("═" * 66 + "\n")
    return state


def run_batch(n: int = 10, depth: int = 5) -> None:
    """
    Run n random games and report the win rate.
    To test for different heuristic strategy and their success rate.
    """
    wins = 0
    for i in range(n):
        print(f"\n{'─'*20}  Game {i+1}/{n}  {'─'*20}")
        state = simulate_game(depth=depth, seed=None)
        if state.check_win():
            wins += 1
    print(f"\n  WIN RATE: {wins}/{n}  ({100*wins/n:.0f}%)\n")



if __name__ == "__main__":
    #simulate_game(depth= 0) # greedy best-first
    #simulate_game(depth =5) # branch and bound
    
    # To benchmark over many games
    run_batch(n=100, depth=0)
