import os
import sys
import math

if 'SUMO_HOME' not in os.environ:
    raise EnvironmentError("未检测到 SUMO_HOME 环境变量，请先配置 SUMO_HOME")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci
import traci.exceptions

from uav import UAV


class UAVEnv:
    def __init__(self, sumocfg_file, uav_start=(1600, 1600),
                 uav_radius=200, step_size=20,
                 x_min=0, x_max=5000, y_min=0, y_max=5000,
                 max_steps=20, gui=True, move_cost=0.0, gui_delay_ms=1000):
        self.sumocfg_file = sumocfg_file
        self.gui = gui
        self.move_cost = move_cost
        self.gui_delay_ms = gui_delay_ms

        self.uav = UAV(
            x=uav_start[0],
            y=uav_start[1],
            radius=uav_radius,
            step_size=step_size,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )

        self.uav_start = uav_start
        self.step_count = 0
        self.max_steps = max_steps
        self.started = False

        self.uav_vehicle_id = "uav"
        self.uav_route_id = "__uav_route__"
        self.uav_vehicle_added = False

    def start_sumo(self, gui=None):
        use_gui = self.gui if gui is None else gui
        sumo_binary = "sumo-gui" if use_gui else "sumo"

        sumo_cmd = [sumo_binary, "-c", self.sumocfg_file]
        if use_gui:
            sumo_cmd.extend(["--start", "--delay", str(self.gui_delay_ms)])

        traci.start(sumo_cmd)
        self.started = True
        self.uav_vehicle_added = False

    def close_sumo(self):
        if self.started:
            traci.close()
            self.started = False
            self.uav_vehicle_added = False

    def reset_uav(self):
        self.uav.reset(self.uav_start[0], self.uav_start[1])
        self.step_count = 0

    def _vehicle_ids_without_uav(self):
        return [
            veh_id for veh_id in traci.vehicle.getIDList()
            if veh_id != self.uav_vehicle_id
        ]

    def _get_display_edge(self):
        for edge_id in traci.edge.getIDList():
            if not edge_id.startswith(":"):
                return edge_id
        return None

    def _ensure_uav_vehicle(self):
        if self.uav_vehicle_added:
            return

        edge_id = self._get_display_edge()
        if edge_id is None:
            return

        route_ids = set(traci.route.getIDList())
        if self.uav_route_id not in route_ids:
            traci.route.add(self.uav_route_id, [edge_id])

        try:
            traci.vehicle.add(
                vehID=self.uav_vehicle_id,
                routeID=self.uav_route_id,
                typeID="DEFAULT_VEHTYPE"
            )
        except traci.exceptions.TraCIException:
            return

        self.uav_vehicle_added = True

        try:
            traci.vehicle.setColor(self.uav_vehicle_id, (255, 0, 0, 255))
            traci.vehicle.setLength(self.uav_vehicle_id, 6.0)
            traci.vehicle.setWidth(self.uav_vehicle_id, 3.0)
            traci.vehicle.setSpeed(self.uav_vehicle_id, 0.0)
        except traci.exceptions.TraCIException:
            pass

    def _sync_uav_vehicle_position(self):
        if not self.started:
            return

        self._ensure_uav_vehicle()
        if self.uav_vehicle_id not in traci.vehicle.getIDList():
            return

        uav_x, uav_y = self.uav.get_position()

        try:
            traci.vehicle.moveToXY(
                vehID=self.uav_vehicle_id,
                edgeID="",
                laneIndex=0,
                x=uav_x,
                y=uav_y,
                keepRoute=2
            )
            traci.vehicle.setSpeed(self.uav_vehicle_id, 0.0)
        except traci.exceptions.TraCIException:
            pass

    def get_vehicle_positions(self):
        vehicle_positions = []
        for veh_id in self._vehicle_ids_without_uav():
            x, y = traci.vehicle.getPosition(veh_id)
            vehicle_positions.append((veh_id, x, y))
        return vehicle_positions

    def count_covered_vehicles(self):
        uav_x, uav_y = self.uav.get_position()
        count = 0

        for veh_id in self._vehicle_ids_without_uav():
            veh_x, veh_y = traci.vehicle.getPosition(veh_id)
            distance = math.hypot(veh_x - uav_x, veh_y - uav_y)

            if distance <= self.uav.radius:
                count += 1

        return count

    def get_state(self):
        return self.uav.get_position()

    def simulation_step(self):
        traci.simulationStep()
        self.step_count += 1
        self._sync_uav_vehicle_position()

    def reset(self):
        if self.started:
            self.close_sumo()

        self.start_sumo(self.gui)
        self.reset_uav()
        self._ensure_uav_vehicle()
        self._sync_uav_vehicle_position()
        return self.get_state()

    def step(self, action):
        self.uav.move(action)
        self._sync_uav_vehicle_position()
        self.simulation_step()

        next_state = self.get_state()
        covered_count = self.count_covered_vehicles()

        if action == 'STAY':
            move_penalty = 0
        else:
            move_penalty = self.move_cost

        reward = covered_count - move_penalty

        done = self.step_count >= self.max_steps
        return next_state, reward, done
