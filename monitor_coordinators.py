"""
Real-time monitoring dashboard for the coordinator cluster.

This script monitors the health and CRDT state of all coordinators
in the cluster, providing a live view of the distributed system.
"""

import requests
import time
import os
import json
from datetime import datetime


# Coordinator URLs to monitor
coordinator_urls = [
    'http://localhost:8001',
    'http://localhost:8002',
    'http://localhost:8003'
]


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_coordinator_status(url: str) -> dict:
    """
    Get the status of a coordinator.
    
    Args:
        url: Base URL of the coordinator
        
    Returns:
        Dictionary with status information
    """
    result = {
        'url': url,
        'online': False,
        'health': None,
        'crdt_summary': None,
        'error': None
    }
    
    try:
        # Check health endpoint
        health_response = requests.get(f"{url}/health", timeout=2)
        if health_response.status_code == 200:
            result['online'] = True
            result['health'] = health_response.json()
        
        # Get CRDT summary
        crdt_response = requests.get(f"{url}/crdt/summary", timeout=2)
        if crdt_response.status_code == 200:
            result['crdt_summary'] = crdt_response.json()
            
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection refused"
    except requests.exceptions.Timeout:
        result['error'] = "Request timeout"
    except requests.exceptions.RequestException as e:
        result['error'] = str(e)
    except json.JSONDecodeError:
        result['error'] = "Invalid JSON response"
    
    return result


def print_coordinator_status(status: dict):
    """
    Print the status of a coordinator.
    
    Args:
        status: Dictionary with coordinator status
    """
    url = status['url']
    online = status['online']
    
    # Print header with status
    status_str = "\033[92mONLINE\033[0m" if online else "\033[91mOFFLINE\033[0m"
    print(f"\n{'='*60}")
    print(f"Coordinator: {url}")
    print(f"Status: {status_str}")
    
    if not online:
        if status['error']:
            print(f"Error: {status['error']}")
        return
    
    # Print health info
    if status['health']:
        health = status['health']
        print(f"Node ID: {health.get('node_id', 'N/A')}")
    
    # Print CRDT summary
    if status['crdt_summary']:
        summary = status['crdt_summary']
        
        print(f"\n--- CRDT State ---")
        print(f"Model Version: {summary.get('model_version', 'N/A')}")
        print(f"Training Rounds: {summary.get('training_rounds', 'N/A')}")
        print(f"Total Samples: {summary.get('total_samples', 'N/A')}")
        
        # Print mood counts
        mood_counts = summary.get('mood_counts', {})
        if mood_counts:
            print(f"\nMood Counts:")
            for mood, count in sorted(mood_counts.items()):
                print(f"  - {mood}: {count}")
        else:
            print(f"\nMood Counts: (none)")
        
        # Print known moods
        known_moods = summary.get('known_moods', [])
        if known_moods:
            print(f"\nKnown Moods: {', '.join(sorted(known_moods))}")


def print_dashboard():
    """Print the monitoring dashboard."""
    clear_screen()
    
    # Print header
    print("=" * 60)
    print("       COORDINATOR CLUSTER MONITOR")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Monitoring {len(coordinator_urls)} coordinators")
    
    # Get and print status for each coordinator
    online_count = 0
    for url in coordinator_urls:
        status = get_coordinator_status(url)
        print_coordinator_status(status)
        if status['online']:
            online_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cluster Health: {online_count}/{len(coordinator_urls)} coordinators online")
    print("=" * 60)
    print("\nPress Ctrl+C to exit")


def main():
    """Main monitoring loop."""
    print("Starting coordinator cluster monitor...")
    print(f"Monitoring URLs: {coordinator_urls}")
    print("Refreshing every 5 seconds...")
    time.sleep(2)
    
    try:
        while True:
            print_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Goodbye!")


if __name__ == "__main__":
    main()
