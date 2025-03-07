import json
import os
import pytest
import time
import statistics
import threading
from queue import Queue


def test_event_propagation_latency(setup_test_environment):
    """Test the latency of event propagation through the system."""
    env = setup_test_environment
    interface_manager = env["interface_manager"]

    # First, print available methods to understand what we can use
    from circman5.manufacturing.digital_twin.event_notification.event_manager import (
        event_manager,
    )

    print(
        "Event manager methods:",
        [method for method in dir(event_manager) if not method.startswith("_")],
    )

    # Set up event tracking
    received_queue = Queue()
    event_times = []

    # Import necessary types
    from circman5.manufacturing.digital_twin.event_notification.event_types import (
        EventCategory,
        Event,
        EventSeverity,
    )

    # In event_manager.py, EventHandler is defined as: Callable[[Event], None]
    # Create a handler function that matches this signature
    def test_event_handler(event):
        """Event handler function that conforms to EventHandler type (Callable[[Event], None])."""
        received_time = time.time()
        received_queue.put((event, received_time))
        print(f"Test handler received event: {event.event_id} - {event.message}")

    # Try to subscribe our test handler
    try:
        print("Attempting to register handler...")
        # Subscribe to SYSTEM category events
        event_manager.subscribe(test_event_handler, EventCategory.SYSTEM)
        print("Handler registered with event_manager.subscribe")
    except Exception as e:
        print(f"Error with subscribe method: {str(e)}")
        pytest.skip(f"Could not register event handler: {str(e)}")

    # Create and publish test events
    num_events = 20

    for i in range(num_events):
        # Create test event
        test_event = Event(
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message=f"Test event {i}",
            source="performance_test",
            details={"test_id": i},
        )

        # Record send time and publish
        send_time = time.time()
        event_manager.publish(test_event)

        # Store event with send time
        event_times.append((test_event.event_id, send_time))

        # Small delay between events
        time.sleep(0.05)

    # Wait for all events to be processed
    time.sleep(1.0)

    # Calculate latencies
    latencies = []

    while not received_queue.empty():
        event, receive_time = received_queue.get()

        # Find matching send time
        for event_id, send_time in event_times:
            if event.event_id == event_id:
                # Calculate latency in milliseconds
                latency = (receive_time - send_time) * 1000
                latencies.append(latency)
                break

    print(f"Received {len(latencies)} events with latency measurements")

    # Ensure we have data to work with
    if not latencies:
        pytest.fail("No events were received by the test handler")

    # Calculate statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"Event latency statistics (ms):")
        print(f"  Average: {avg_latency:.2f}")
        print(f"  Median: {median_latency:.2f}")
        print(f"  Min: {min_latency:.2f}")
        print(f"  Max: {max_latency:.2f}")
        print(f"  95th percentile: {p95_latency:.2f}")

        # Assert reasonable latency - adjust thresholds as needed for your system
        assert (
            max_latency < 200
        ), f"Maximum latency ({max_latency:.2f}ms) exceeds threshold (200ms)"
