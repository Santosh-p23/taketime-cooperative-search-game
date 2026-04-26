"""
FastAPI server for Take Time game state solver.
Provides a POST endpoint to get the next best move from a given game state.

Usage:
    uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from take_time import GameState
from state_solver import get_best_move, upper_bound, LOSS_SCORE

app = FastAPI(title="Take Time Solver API")


# ══════════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ══════════════════════════════════════════════════════════════════════════════

class CardModel(BaseModel):
    value: int
    color: str


class GameStateRequest(BaseModel):
    hands: List[List[List[int | str]]]  # e.g., [[[8, "W"]], [[12, "B"]], [[10, "B"], [2, "B"]]]
    zones: Dict[str, List[List[int | str]]]  # e.g., {"1": [[5, "W"]], "2": [], ...}
    turn: int


class CardResponse(BaseModel):
    value: int
    color: str


class MoveResponse(BaseModel):
    verdict: str
    card: Optional[CardResponse] = None
    zone: Optional[int] = None
    reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_card(card_data: List[int | str]) -> tuple[int, str]:
    """Convert [value, color] to (value, color) tuple."""
    return int(card_data[0]), str(card_data[1])


def request_to_gamestate(request: GameStateRequest) -> GameState:
    """Convert API request to GameState object."""
    hands = [
        [parse_card(c) for c in hand]
        for hand in request.hands
    ]

    state = GameState(hands)
    state.zones = {
        int(z): [parse_card(c) for c in cards]
        for z, cards in request.zones.items()
    }
    state.turn = request.turn
    return state


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Take Time Solver API", "endpoint": "/solve"}


@app.post("/solve", response_model=MoveResponse)
def solve_game_state(request: GameStateRequest) -> MoveResponse:
    """
    Given a game state, return the next best move.
    
    Request body:
    {
        "hands": [[[value, "color"], ...], ...],
        "zones": {"1": [[value, "color"], ...], "2": ..., ...},
        "turn": int
    }
    
    Response:
    {
        "verdict": "OK" | "LOSS",
        "card": {"value": int, "color": str},
        "zone": int,
        "reason": str (if loss)
    }
    """
    try:
        state = request_to_gamestate(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid game state: {str(e)}")

    # Check for legal moves
    if not state.get_legal_moves():
        return MoveResponse(
            verdict="LOSS",
            reason="No legal moves available"
        )

    # Check if state is winnable
    if upper_bound(state) <= LOSS_SCORE:
        return MoveResponse(
            verdict="LOSS",
            reason="Game state is unwinnable"
        )

    # Get best move (depth 4 for strong play)
    move = get_best_move(state, depth=4)

    if move is None:
        return MoveResponse(
            verdict="LOSS",
            reason="Solver could not find a move"
        )

    card, zone = move
    return MoveResponse(
        verdict="OK",
        card=CardResponse(value=card[0], color=card[1]),
        zone=zone
    )


@app.post("/solve/depth/{depth}", response_model=MoveResponse)
def solve_game_state_depth(
    request: GameStateRequest,
    depth: int = 4
) -> MoveResponse:
    """
    Given a game state and solver depth, return the next best move.
    Higher depth = stronger play but slower.
    """
    if depth < 0 or depth > 6:
        raise HTTPException(status_code=400, detail="Depth must be between 0 and 6")

    try:
        state = request_to_gamestate(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid game state: {str(e)}")

    if not state.get_legal_moves():
        return MoveResponse(
            verdict="LOSS",
            reason="No legal moves available"
        )

    if upper_bound(state) <= LOSS_SCORE:
        return MoveResponse(
            verdict="LOSS",
            reason="Game state is unwinnable"
        )

    move = get_best_move(state, depth=depth)

    if move is None:
        return MoveResponse(
            verdict="LOSS",
            reason="Solver could not find a move"
        )

    card, zone = move
    return MoveResponse(
        verdict="OK",
        card=CardResponse(value=card[0], color=card[1]),
        zone=zone
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)