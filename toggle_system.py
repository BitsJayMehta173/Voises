import psutil
import json
import os

PID_FILE = "pids.json"
STATE_FILE = "system_state.json"

def toggle():
    if not os.path.exists(PID_FILE):
        print("System is not running! Start it first with 'python run_all.py'.")
        return

    with open(PID_FILE, "r") as f:
        pids = json.load(f)

    # Determine current state
    current_state = "playing"
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state_data = json.load(f)
            current_state = state_data.get("state", "playing")

    if current_state == "playing":
        print("Pausing System (Suspending processes)...")
        # Suspend the recorder and processor
        for name, pid in pids.items():
            try:
                p = psutil.Process(pid)
                p.suspend()
                print(f"  - Paused {name} (PID: {pid})")
            except Exception as e:
                print(f"  - Error pausing {name}: {e}")
        
        with open(STATE_FILE, "w") as f:
            json.dump({"state": "paused"}, f)
        print("\n[PAUSED] The system is now waiting in the background. It won't use CPU.")

    else:
        print("Resuming System (Playing processes)...")
        # Resume the recorder and processor
        for name, pid in pids.items():
            try:
                p = psutil.Process(pid)
                p.resume()
                print(f"  - Resumed {name} (PID: {pid})")
            except Exception as e:
                print(f"  - Error resuming {name}: {e}")
        
        with open(STATE_FILE, "w") as f:
            json.dump({"state": "playing"}, f)
        print("\n[PLAYING] The system is live and listening again!")

if __name__ == "__main__":
    toggle()
