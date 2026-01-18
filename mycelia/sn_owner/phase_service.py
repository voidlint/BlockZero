import bittensor
import uvicorn
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Query

from mycelia.shared.config import OwnerConfig, parse_args
from mycelia.shared.rate_limiter import get_rate_limiter
from mycelia.shared.signature_utils import sign_data
from mycelia.sn_owner.cycle import PhaseManager, PhaseResponse
from mycelia.sn_owner.expert_assignment_manager import ExpertAssignmentManager


# Thread-safe state container
class AppState:
    """Thread-safe application state container."""

    def __init__(self):
        self.config: OwnerConfig | None = None
        self.phase_manager: PhaseManager | None = None
        self.assignment_manager: ExpertAssignmentManager | None = None
        self.wallet: bittensor.Wallet | None = None


# Global state instance (thread-safe via FastAPI dependency injection)
_app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup app state."""
    # Startup: state already initialized in main()
    yield
    # Shutdown: cleanup if needed
    pass


app = FastAPI(title="Phase Service", lifespan=lifespan)


# Dependency injection for thread-safe state access
def get_state() -> AppState:
    """Get application state (thread-safe via FastAPI)."""
    return _app_state


@app.get("/get_phase", response_model=PhaseResponse)
async def read_phase(state: AppState = Depends(get_state)):
    """
    Returns which phase we're in for the given block height.
    """
    try:
        if state.phase_manager is None:
            raise HTTPException(status_code=503, detail="Phase manager not initialized")
        return state.phase_manager.get_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/previous_phase_blocks", response_model=dict[str, tuple[int, int]])
async def prev_phase(state: AppState = Depends(get_state)):
    """
    Returns which phase we're in for the given block height.
    """
    try:
        if state.phase_manager is None:
            raise HTTPException(status_code=503, detail="Phase manager not initialized")
        return state.phase_manager.previous_phase_block_ranges()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/blocks_until_next_phase", response_model=dict[str, int])
async def next_phase(state: AppState = Depends(get_state)):
    """
    Returns which phase we're in for the given block height.
    """
    try:
        if state.phase_manager is None:
            raise HTTPException(status_code=503, detail="Phase manager not initialized")
        return state.phase_manager.blocks_until_next_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get-expert-assignment")
async def get_expert_assignment(
    miner_hotkey: str = Query(..., description="Miner's SS58 address"),
    expert_group_id: int = Query(..., description="Expert group ID (0=math, 1=agentic, 2=planning, etc.)"),
    state: AppState = Depends(get_state),
):
    """
    Returns the authoritative expert assignment for a specific miner.
    
    This enforces centralized control - miners MUST use the assignment
    provided by the SN owner. Prevents miners from self-selecting experts.
    
    Args:
        miner_hotkey: Miner's SS58 address
        expert_group_id: Which expert group (0=math, 1=agentic, 2=planning, 3=vision, etc.)
        
    Returns:
        {
            "miner_hotkey": str,
            "expert_group_id": int,
            "layer_assignments": {layer_id: [(my_expert_id, org_expert_id), ...]},
            "timestamp": float,
        }
        
    Raises:
        403: Miner not authorized for this expert group
        404: No assignment found
    """
    # ✅ SECURITY: Rate limiting
    rate_limiter = get_rate_limiter("ping")
    allowed, reason = rate_limiter.is_allowed(miner_hotkey)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    try:
        if state.assignment_manager is None:
            raise HTTPException(status_code=503, detail="Assignment manager not initialized")
        if state.wallet is None:
            raise HTTPException(status_code=503, detail="Wallet not initialized")

        assignment = state.assignment_manager.get_assignment(miner_hotkey, expert_group_id)

        if assignment is None:
            raise HTTPException(
                status_code=403,
                detail=f"Miner {miner_hotkey} not authorized for expert group {expert_group_id}. "
                       "Contact subnet owner for assignment."
            )

        import time

        # Build unsigned response
        response_data = {
            "miner_hotkey": miner_hotkey,
            "expert_group_id": expert_group_id,
            "layer_assignments": {
                str(layer_id): mappings  # JSON requires string keys
                for layer_id, mappings in assignment.items()
            },
            "timestamp": time.time(),
            "sn_owner_hotkey": state.wallet.hotkey.ss58_address,
        }

        # ✅ SECURITY: Sign the assignment with SN owner's hotkey
        signature = sign_data(state.wallet, response_data)
        response_data["signature"] = signature
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Phase service is running",
        "cycle_length": phase_manager.cycle_length,
        "phases": [{"index": i, "name": p["name"], "length": p["length"]} for i, p in enumerate(phase_manager.phases)],
        "endpoints": [
            "GET /get_phase",
            "GET /previous_phase_blocks",
            "GET /blocks_until_next_phase",
            "GET /get-expert-assignment?miner_hotkey=...&expert_group_id=...",
        ],
        "usage": "GET /phase?block_height=123",
    }


if __name__ == "__main__":
    args = parse_args()

    # Initialize config
    if args.path:
        config = OwnerConfig.from_path(args.path)
    else:
        config = OwnerConfig()

    config.write()

    # Load wallet for signing
    wallet = bittensor.Wallet(
        name=config.wallet_name,
        hotkey=config.wallet_hotkey_name,
    )

    subtensor = bittensor.Subtensor(network=config.chain.network)

    # Initialize app state (thread-safe via FastAPI dependency injection)
    _app_state.config = config
    _app_state.phase_manager = PhaseManager(config, subtensor)
    _app_state.assignment_manager = ExpertAssignmentManager(config)
    _app_state.wallet = wallet

    uvicorn.run(app, host=config.owner.app_ip, port=config.owner.app_port)
