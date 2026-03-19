Project: SUMO traffic simulation with TraCI (Python)

This repository contains SUMO network/route/config files and a small Python controller using TraCI.

Quick architecture
- **Simulation data**: network and route definitions live as XML files at the repo root: `jnu_clean.net.xml`, `jnu_clean.rou.xml`, and `jnu_clean.sumocfg`.
- **Controller**: `test_traci.py` starts/controls SUMO via TraCI (`traci.start([...])`) and runs a simple simulation loop.
- **Why this layout**: separation of static inputs (network/routes/config) from dynamic logic (Python TraCI script) keeps simulation scenarios easy to swap by changing the XMLs or `.sumocfg`.

Developer prerequisites
- Install SUMO (https://www.eclipse.org/sumo/) and ensure SUMO binaries are on `PATH` or set `SUMO_HOME` environment variable.
- Make the TraCI Python bindings available to your Python environment. Two common options on Windows:
  - Add `%SUMO_HOME%\tools` to `PYTHONPATH` (recommended when using the SUMO-provided `traci`), or
  - If available for your platform, install the `traci` package via `pip install traci`.

Example run (PowerShell)
```powershell
# If SUMO is on PATH or SUMO_HOME is set
python test_traci.py

# Headless (no GUI): edit test_traci.py to set sumoBinary = "sumo"
# Or run SUMO manually and connect remotely (not required by current test_traci.py):
sumo -c jnu_clean.sumocfg --start --remote-port 8873
python test_traci.py  # if script configured to connect to that port
```

Project-specific patterns and conventions
- `*.net.xml` — SUMO network definition. Changes here alter map topology.
- `*.rou.xml` — route files / vehicle flows. Edit to change traffic demand.
- `*.sumocfg` — scenario configuration; this is the file passed to SUMO (`-c`).
- `test_traci.py` shows how this repo expects to start SUMO from Python using `traci.start([sumoBinary, "-c", sumoConfig])` and iterate via `traci.simulationStep()`.

Debugging and iteration notes
- Running inside `Program Files (x86)` can trigger permission issues on Windows; prefer a user-writable workspace when iterating.
- For quick inspection, run `sumo-gui -c jnu_clean.sumocfg` to visualize vehicle movements.
- To increase simulation steps or instrument outputs, modify the loop in `test_traci.py` (it currently runs 100 steps).

Integration points and external dependencies
- SUMO binaries (external, native). Ensure `sumo`/`sumo-gui` executable is reachable.
- TraCI Python bindings (from SUMO `tools` or PyPI `traci`). The code imports `traci` directly.

What to look for when making changes
- If you add a new scenario, include a new `.sumocfg` that references the correct `.net.xml` and `.rou.xml` files.
- Keep simulation logic in Python small and testable: e.g., extract vehicle-processing logic from the main loop into importable functions for unit testing.

Examples referenced in this repo
- Controller example: `test_traci.py` (starts SUMO via `traci.start` and loops `traci.simulationStep()`).
- Config example: `jnu_clean.sumocfg` (used by `test_traci.py`).

Notes for AI agents
- Prefer small, targeted edits: this repo is scenario-driven; changing XMLs changes the whole simulation.
- When adding instructions or automation, include explicit `sumo`/`sumo-gui` commands and whether `PYTHONPATH` must include `%SUMO_HOME%\tools`.
- Do not assume a CI or test harness exists — propose lightweight runners (e.g., a `run.sh`/`run.ps1`) if asked.

If anything in this file is unclear or you want more examples (e.g., how `traci.vehicle.getPosition` is used), tell me which area to expand.
