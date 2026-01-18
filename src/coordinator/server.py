from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
import requests
import json
import time

from src.crdt.state_manager import CRDTStateManager
from src.coordinator.aggregator import federated_average, deserialize_model_state, serialize_model_state
from src.models.mood_classifier import create_model


# Global variables
app = FastAPI()
coordinator_state = None  # Will be CRDTStateManager
pending_updates = []  # List to store platform updates
global_model = None  # The aggregated model
other_coordinator_urls = []  # List of other coordinator URLs

# Configuration (set via command line args)
node_id = None
port = None


# Pydantic models
class PlatformUpdate(BaseModel):
    platform_id: int
    model_state: dict
    n_samples: int
    local_rounds: int
    model_version: int


class CRDTSyncRequest(BaseModel):
    crdt_state: dict


@app.on_event("startup")
async def startup_event():
    """Initialize coordinator state on startup."""
    global coordinator_state, global_model, node_id, port
    
    # Initialize CRDT state manager with node_id
    if coordinator_state is None:
        coordinator_state = CRDTStateManager(node_id=node_id)
    
    # Initialize global model
    if global_model is None:
        global_model = create_model()
    
    print(f"Coordinator {node_id} started on port {port}")


# Internal functions
def sync_with_coordinators():
    """Sync CRDT state with other coordinators."""
    if not other_coordinator_urls:
        return
    
    # Serialize coordinator state
    crdt_state = coordinator_state.serialize()
    
    for url in other_coordinator_urls:
        try:
            response = requests.post(
                f"{url}/crdt/sync",
                json={"crdt_state": crdt_state},
                timeout=10
            )
            if response.status_code == 200:
                print(f"Successfully synced with {url}")
            else:
                print(f"Failed to sync with {url}: {response.status_code}")
        except requests.RequestException as e:
            print(f"Failed to sync with {url}: {e}")


def perform_aggregation():
    """Perform federated averaging when enough updates are collected."""
    global pending_updates, global_model
    
    if len(pending_updates) < 5:
        return
    
    print(f"Performing aggregation with {len(pending_updates)} updates...")
    
    # Perform federated averaging
    aggregated_state = federated_average(pending_updates)
    
    # Deserialize and load into global model
    tensor_state = deserialize_model_state(aggregated_state)
    global_model.load_state_dict(tensor_state)
    
    # Update coordinator state
    coordinator_state.update_model_version()
    coordinator_state.increment_training_rounds()
    
    # Update mood counts based on pending updates
    for update in pending_updates:
        coordinator_state.add_samples(update['n_samples'])
    
    # Clear pending updates
    pending_updates = []
    
    # Sync with other coordinators
    sync_with_coordinators()
    
    print(f"Aggregation complete. Model version: {coordinator_state.get_model_version()}")
    print(f"Summary: {coordinator_state.get_summary()}")


# FastAPI Endpoints
@app.post("/platform/update")
async def receive_platform_update(update: PlatformUpdate):
    """Receive model update from a platform."""
    global pending_updates
    
    # Append to pending updates
    pending_updates.append({
        'platform_id': update.platform_id,
        'model_state': update.model_state,
        'n_samples': update.n_samples,
        'local_rounds': update.local_rounds,
        'model_version': update.model_version
    })
    
    # Increment total samples in coordinator state
    coordinator_state.increment_total_samples(update.n_samples)
    
    print(f"Received update from platform {update.platform_id}, pending: {len(pending_updates)}")
    
    # Trigger aggregation if we have enough updates
    if len(pending_updates) >= 5:
        perform_aggregation()
    
    return {"status": "received", "pending": len(pending_updates)}


@app.get("/model/latest")
async def get_latest_model():
    """Return the current global model state."""
    return {
        "model_state": serialize_model_state(global_model.state_dict()),
        "model_version": coordinator_state.get_model_version()
    }


@app.post("/crdt/sync")
async def sync_crdt(request: CRDTSyncRequest):
    """Receive and merge CRDT state from another coordinator."""
    try:
        # Deserialize and merge CRDT state
        coordinator_state.merge(request.crdt_state)
        return {"status": "synced"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/crdt/summary")
async def get_crdt_summary():
    """Return CRDT state summary."""
    return coordinator_state.get_summary()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "node_id": node_id, "port": port}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinator Server")
    parser.add_argument("--node-id", type=str, default="coordinator-1", help="Unique node ID")
    parser.add_argument("--port", type=int, default=8001, help="Port to run server on")
    
    args = parser.parse_args()
    
    node_id = args.node_id
    port = args.port
    
    # Initialize globals before starting server
    coordinator_state = CRDTStateManager(node_id)
    global_model = create_model()
    
    # Set other coordinator URLs (hardcode for now)
    all_ports = [8001, 8002, 8003]
    other_coordinator_urls = [f"http://localhost:{p}" for p in all_ports if p != port]
    
    uvicorn.run(app, host="0.0.0.0", port=port)
