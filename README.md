# Take Time Cooperative Game Solver

This repository implements a cooperative solver for a 3-player card-placement game.

## Project Files

- `take_time.py`: core game model, heuristics, search, simulation, and runner.
- `state_solver.py`: state JSON loader and helper API for next best move and simulation.
- `heuristic_learner.py`: machine learning module to learn optimal heuristic weights from simulated games using logistic regression.
- `gamestate_example.json`: example input state, if present (used by CLI default).

## Game Rules Summary

- 3 players, 4 cards each (12 total).
- 6 zones (1..6), must place all 12 cards one per turn.
- Zone constraints:
  - Zone 1: only white (`W`) cards.
  - Zone 6: at least 3 cards by game end.
  - All zones must be non-empty by end.
  - Zone sums must be non-decreasing from 1 to 6.
  - Each zone sum ≤ 24.

## Key classes and functions

### `take_time.py`

- `GameState`:
  - `hands`, `zones`, `turn`, `move_history`
  - `current_player`, `is_terminal`, `cards_remaining`, `zone_sums`, `zone_counts`, `all_remaining_values`
  - `is_legal_move(card, zone)`, `get_legal_moves()`, `apply_move(card, zone)`
  - `check_win()` verifies conditions.

- `heuristic(state)` returns non-terminal score for solver guidance.
- `upper_bound(state)` bounds branch value for pruning.
- `order_moves(state, moves)` best-first on `heuristic`.
- `cooperative_search(state, depth, alpha, nodes_visited)` uses branch-and-bound and alpha updates.
- `get_best_move(state, depth=5)` returns next best move and stats.

- `simulate_game(depth=5, seed=None)`: self-play interactive stdout simulation.
- `run_batch(n=10, depth=5)`: evaluate win rate.

### `state_solver.py`

- `load_state(filepath)`: loads JSON game state and returns `GameState`.
- `next_best_move(state)`: pass/fail verdict, `zone`, `card`.
- `simulate_from_state(state, depth=4)`: simulate until end with move history and final result.

### `heuristic_learner.py`

Machine learning module to optimize heuristic weights via logistic regression:

- `extract_features(state)`: converts game state to 13-element feature vector (ordering, coverage, zone progress, penalties, etc.).
- `simulate_one_game(depth)`: plays one full game, returns feature vectors + win/loss label.
- `collect_data(n_games, depth)`: runs N simulations to build training dataset.
- `train_and_extract_weights(X, y)`: trains logistic regression, returns learned weights and accuracy.
- `make_learned_heuristic(weights)`: wraps learned weights as a drop-in heuristic function for `cooperative_search`.
- `benchmark(weights, n_eval, depth)`: compares win rates between original and learned heuristic.

## Usage

1. Run built-in simulation:

```bash
python take_time.py
```

2. Evaluate a specific state JSON:

```bash
python state_solver.py gamestate_example.json
```

3. Train a learned heuristic:

```bash
python heuristic_learner.py                  # train with defaults (1000 games)
python heuristic_learner.py --games 2000     # increase training data
python heuristic_learner.py --depth 2        # stronger solver during data collection
python heuristic_learner.py --no-bench       # skip benchmark (faster)
```

4. Use in code:

```python
from state_solver import load_state, next_best_move, simulate_from_state
state = load_state('gamestate_example.json')
print(next_best_move(state))
print(simulate_from_state(state, depth=5))
```

## Development notes

- Increase `depth` to strengthen planning (slower search, better play).
- Adjust heuristic scoring in `take_time.py` for strategy tuning.
