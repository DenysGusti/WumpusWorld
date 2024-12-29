import heapq
from collections import deque
from itertools import product
from functools import reduce
from operator import mul
from wumpus import Orientation, Actions, Percepts


class KnowledgeBase:
    def __init__(self, size: tuple[int, int]) -> None:
        self.stench: list[list[bool]] = [[False for _ in range(size[1])] for _ in range(size[0])]
        self.breeze: list[list[bool]] = [[False for _ in range(size[1])] for _ in range(size[0])]
        # 1 - known, 0 - frontier, -1 - other
        self.cell_status: list[list[int]] = [[-1 for _ in range(size[1])] for _ in range(size[0])]
        self.cell_status[0][0] = 0
        self.frontier: set[tuple[int, int]] = {(0, 0)}
        self.wumpus: tuple[int, int] | None = None
        self.stench_frontier: set[tuple[int, int]] = set()
        self.breeze_frontier: set[tuple[int, int]] = set()
        self.confident_pits: set[tuple[int, int]] = set()
        self.has_arrow = True
        self.p_pit: float = 0.2
        self.p_crit: float = 0.21

    def add_percepts_to_cell(self, cell_position: tuple[int, int],
                             percept: tuple[bool, bool, bool, bool, bool]) -> None:
        x, y = cell_position
        if percept[Percepts.STENCH]:
            self.stench[x][y] = True
            self.stench_frontier.add(cell_position)
        if percept[Percepts.BREEZE]:
            self.breeze[x][y] = True
            self.breeze_frontier.add(cell_position)

    def add_cell_to_known(self, cell_position: tuple[int, int]) -> None:
        x, y = cell_position
        if self.cell_status[x][y] == 1:
            return

        self.cell_status[x][y] = 1
        if cell_position in self.frontier:
            self.frontier.remove(cell_position)
        adjacent_unknown_cells: list[tuple[int, int]] = self.get_adjacent_cells((x, y), -1)
        for cell in adjacent_unknown_cells:
            self.cell_status[cell[0]][cell[1]] = 0
            self.frontier.add(cell)

    def find_wumpus(self) -> None:
        if self.wumpus is not None or len(self.stench_frontier) == 0:
            return
        if len(self.stench_frontier) == 1:
            stench_cell = list(self.stench_frontier)[0]
            adjacent_frontier_cells: list[tuple[int, int]] = self.get_adjacent_cells(stench_cell, 0)
            if len(adjacent_frontier_cells) == 1:
                self.wumpus = adjacent_frontier_cells[0]
            else:
                return
        else:
            stench_frontier_list: list[tuple[int, int]] = list(self.stench_frontier)
            x_a, y_a = stench_frontier_list[0]
            x_b, y_b = stench_frontier_list[1]
            if self.cell_status[x_a][y_b] == 1 and self.cell_status[x_b][y_a] == 0:
                self.wumpus = x_b, y_a
            elif self.cell_status[x_a][y_b] == 0 and self.cell_status[x_b][y_a] == 1:
                self.wumpus = x_a, y_b
            elif x_a == x_b:
                self.wumpus = x_a, (y_a + y_b) // 2
            elif y_a == y_b:
                self.wumpus = (x_a + x_b) // 2, y_a
            else:
                return
        # print(f"{self.wumpus} is a wumpus")

    def all_adjacent_known_stench(self, cell_position: tuple[int, int]) -> bool:
        adjacent_known_cells: list[tuple[int, int]] = self.get_adjacent_cells(cell_position, 1)
        return all(self.stench[x][y] for x, y in adjacent_known_cells)

    def all_adjacent_known_breeze(self, cell_position: tuple[int, int]) -> bool:
        adjacent_known_cells: list[tuple[int, int]] = self.get_adjacent_cells(cell_position, 1)
        return all(self.breeze[x][y] for x, y in adjacent_known_cells)

    def get_closest_promising_cell(self, cell_position: tuple[int, int],
                                   start_orientation: int) -> tuple[int, int] | None:
        distances: dict[tuple[int, int], int] = {
            cell: self.route_to_cell(cell_position, cell, start_orientation, False)[1] for cell in self.frontier}
        sorted_frontier: list[tuple[int, int]] = list(self.frontier)
        sorted_frontier.sort(key=lambda pos: distances[pos])
        for cell in sorted_frontier:
            if self.wumpus is not None and self.wumpus == cell and self.has_arrow:
                continue
            # frontier cell is not a pit and not a wumpus
            if not self.all_adjacent_known_breeze(cell) and not (
                    self.has_arrow and self.all_adjacent_known_stench(cell)):
                return cell

        return None

    def get_adjacent_cells(self, cell_position: tuple[int, int], cell_type: int) -> list[tuple[int, int]]:
        x, y = cell_position
        result: list[tuple[int, int]] = []
        if x > 0 and self.cell_status[x - 1][y] == cell_type:
            result.append((x - 1, y))
        if y > 0 and self.cell_status[x][y - 1] == cell_type:
            result.append((x, y - 1))
        if x < len(self.stench) - 1 and self.cell_status[x + 1][y] == cell_type:
            result.append((x + 1, y))
        if y < len(self.stench[0]) - 1 and self.cell_status[x][y + 1] == cell_type:
            result.append((x, y + 1))
        return result

    def check_pit_frontier_consistency(self, frontier_model: dict[tuple[int, int], bool]) -> bool:
        for breeze_cell in self.breeze_frontier:
            adjacent_frontier_cells: list[tuple[int, int]] = self.get_adjacent_cells(breeze_cell, 0)
            if not any(cell in frontier_model for cell in adjacent_frontier_cells):
                continue
            if not any(frontier_model[cell] for cell in adjacent_frontier_cells if cell in frontier_model):
                return False
        return True

    def evaluate_pit_probability(self, pit_cell: tuple[int, int],
                                 frontier_models: list[dict[tuple[int, int], bool]]) -> float:
        true_models: list[dict[tuple[int, int], bool]] = [model for model in frontier_models if model[pit_cell]]
        false_models: list[dict[tuple[int, int], bool]] = [model for model in frontier_models if not model[pit_cell]]
        true_p: float = sum(
            reduce(mul, [self.p_pit if pit_present else 1 - self.p_pit for pit_present in model.values()], 1.) for model
            in true_models)
        false_p: float = sum(
            reduce(mul, [self.p_pit if pit_present else 1 - self.p_pit for pit_present in model.values()], 1.) for model
            in false_models)
        return true_p / (true_p + false_p)

    def get_closest_safest_pit_cell(self, cell_position: tuple[int, int],
                                    start_orientation: int) -> tuple[int, int] | None:
        if len(self.frontier) == 0:
            return None

        frontier_models: list[dict[tuple[int, int], bool]] = [{cell: pit for cell, pit in zip(self.frontier, model)} for
                                                              model in
                                                              product([False, True], repeat=len(self.frontier))]
        consistent_frontier_models: list[dict[tuple[int, int], bool]] = list(
            filter(self.check_pit_frontier_consistency, frontier_models))

        pit_probabilities: dict[tuple[int, int], float] = {
            cell: self.evaluate_pit_probability(cell, consistent_frontier_models) for cell in self.frontier}
        self.confident_pits: set[tuple[int, int]] = set(cell for cell, p in pit_probabilities.items() if p == 1.)
        probable_pits: list[tuple[tuple[int, int], float]] = list(
            sorted(((cell, p) for cell, p in pit_probabilities.items() if p < 1.), key=lambda item: item[1]))

        if len(probable_pits) == 0 or probable_pits[0][1] >= self.p_crit:
            return None

        safest_pits: list[tuple[int, int]] = [cell for cell, p in probable_pits if p == probable_pits[0][1]]
        distances: dict[tuple[int, int], int] = {
            cell: self.route_to_cell(cell_position, cell, start_orientation, False)[1] for cell in safest_pits}
        safest_pits.sort(key=lambda pos: distances[pos])

        for x, y in safest_pits:
            # wumpus wasn't missed
            if self.cell_status[x][y] != 2:
                return x, y
        return None

    def shoot_possible_pit(self, cell_position: tuple[int, int]) -> bool:
        return (self.wumpus is not None and self.wumpus == cell_position or
                self.wumpus is None and self.all_adjacent_known_stench(cell_position))

    @staticmethod
    def heuristic(from_cell: tuple[int, int], to_cell: tuple[int, int]) -> int:
        return abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])

    @staticmethod
    def reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]],
                         goal: tuple[int, int]) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def wumpus_is_not_in_pit(self) -> bool:
        return not all(stench_cell in self.breeze_frontier for stench_cell in self.stench_frontier)

    @staticmethod
    def output_orientations(orientations: list[list[int]], start: tuple[int, int], end: tuple[int, int],
                            cost: int, path: list[tuple[int, int]]) -> None:
        print(f"{start = }, {end = }, {cost = }, {path = }")
        result = ""
        for row in orientations:
            for cell in row:
                match cell:
                    # maps are rotated 90 degrees clockwise
                    case Orientation.NORTH:
                        result += "> "
                    case Orientation.EAST:
                        result += "v "
                    case Orientation.SOUTH:
                        result += "< "
                    case Orientation.WEST:
                        result += "^ "
                    case 4:
                        result += "- "
            result += '\n'
        print(result)

    @staticmethod
    def get_adjacent_actions(start: tuple[int, int], end: tuple[int, int],
                             orientation: int) -> tuple[list[Actions], int]:
        d_pos = end[0] - start[0], end[1] - start[1]
        turns: int = 0
        if d_pos[0] == -1:
            turns = (Orientation.WEST - orientation) % 4
            orientation = Orientation.WEST
        elif d_pos[0] == 1:
            turns = (Orientation.EAST - orientation) % 4
            orientation = Orientation.EAST
        elif d_pos[1] == -1:
            turns = (Orientation.SOUTH - orientation) % 4
            orientation = Orientation.SOUTH
        elif d_pos[1] == 1:
            turns = (Orientation.NORTH - orientation) % 4
            orientation = Orientation.NORTH

        result: list[Actions] = [Actions.RIGHT for _ in range(turns)] if turns <= 2 else [Actions.LEFT]
        return result, orientation

    # A*-search
    def route_to_cell(self, start: tuple[int, int], goal: tuple[int, int],
                      start_orientation: int, safe: bool) -> [list[tuple[int, int]], int]:
        frontier: list[tuple[int, tuple[int, int]]] = [(self.heuristic(start, goal), start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_costs: dict[tuple[int, int], int] = {start: 0}
        size: tuple[int, int] = len(self.cell_status), len(self.cell_status[0])
        orientations: list[list[int]] = [[4 for _ in range(size[1])] for _ in range(size[0])]
        orientations[start[0]][start[1]] = start_orientation

        while frontier:
            current_cost, current_cell = heapq.heappop(frontier)
            if current_cell == goal:
                path: list[tuple[int, int]] = self.reconstruct_path(came_from, goal)
                # self.output_orientations(orientations, start, goal, g_costs[goal], path)
                return path, g_costs[goal]

            neighbors: list[tuple[int, int]] = self.get_adjacent_cells(current_cell, 1)
            if not safe:
                neighbors += self.get_adjacent_cells(current_cell, 0)
            for neighbor in neighbors:
                current_orientation = orientations[current_cell[0]][current_cell[1]]
                turns, new_orientation = self.get_adjacent_actions(current_cell, neighbor, current_orientation)
                tentative_g_cost = g_costs[current_cell] + len(turns) + 1

                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current_cell
                    g_costs[neighbor] = tentative_g_cost
                    orientations[neighbor[0]][neighbor[1]] = new_orientation
                    priority = tentative_g_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))

        raise ValueError("No path found from start to goal.")

    def __str__(self) -> str:
        result: str = "Stench:\n"
        for row in self.stench:
            for cell in row:
                result += f"{int(cell)} "
            result += '\n'
        result += "Breeze:\n"
        for row in self.breeze:
            for cell in row:
                result += f"{int(cell)} "
            result += '\n'
        result += "Status:\n"
        for i in range(len(self.cell_status)):
            for j in range(len(self.cell_status[0])):
                cell = self.cell_status[i][j]
                result += f"{"W" if cell == 2 else "K" if cell == 1 else (
                    "P" if (i, j) in self.confident_pits else "F") if cell == 0 else "O"} "
            result += '\n'
        result += f"Frontier:\n{self.frontier}\n"
        return result


class Agent:
    def __init__(self, size: tuple[int, int] = (4, 4)) -> None:
        self.size: tuple[int, int] = size
        self.lucky_shoot_idx: list[int] = [0, 1]
        self.t_max = 50

    def new_episode(self) -> None:
        self.kb = KnowledgeBase(self.size)
        self.pos: tuple[int, int] = (0, 0)
        # maps are rotated 90 degrees clockwise
        self.orientation: int = Orientation.NORTH
        self.plan: deque[Actions] = deque()
        self.last_action: Actions = Actions.CLIMB
        self.shoot_pit: tuple[int, int] | None = None
        self.t = 0

    def translate_route_into_actions(self, route: list[tuple[int, int]]) -> list[Actions]:
        result: list[Actions] = []
        for cell in route:
            actions, self.orientation = self.kb.get_adjacent_actions(self.pos, cell, self.orientation)
            result += actions
            result.append(Actions.FORWARD)
            self.pos = cell
        return result

    def get_closest_adjacent_stench_cell(self, stench_point: tuple[int, int]) -> tuple[int, int]:
        adjacent_stench_frontier: list[tuple[int, int]] = self.kb.get_adjacent_cells(stench_point, 0)
        if len(adjacent_stench_frontier) == 0:
            raise ValueError("Very bad.")

        distances: dict[tuple[int, int], int] = {
            cell: len(self.kb.get_adjacent_actions(stench_point, cell, self.orientation)) for cell in
            adjacent_stench_frontier}
        adjacent_stench_frontier.sort(key=lambda pos: distances[pos])
        return adjacent_stench_frontier[0]

    def kill_wumpus(self, pit: tuple[int, int] | None = None) -> None:
        if not self.kb.has_arrow:
            return

        self.kb.has_arrow = False
        if self.kb.wumpus is not None:
            self.kb.add_cell_to_known(self.kb.wumpus)
            self.go_to_cell(self.kb.wumpus)
            if len(self.plan) > 0:
                self.plan.pop()
            self.plan += [Actions.SHOOT, Actions.FORWARD]
        else:
            # shoot and kill 50/50 or 33/66
            stench_point = list(self.kb.stench_frontier)[0]
            self.go_to_cell(stench_point)
            cell = self.get_closest_adjacent_stench_cell(stench_point) if pit is None else pit
            actions, self.orientation = self.kb.get_adjacent_actions(self.pos, cell, self.orientation)
            self.plan += actions
            self.plan.append(Actions.SHOOT)

    def go_to_cell(self, cell_position: tuple[int, int]) -> None:
        route, cost = self.kb.route_to_cell(self.pos, cell_position, self.orientation, True)
        self.plan += self.translate_route_into_actions(route)

    def reconfigure_path(self, current_pos: tuple[int, int], current_orientation: int) -> None:
        plan = self.plan.copy()
        goal_pos, goal_orientation = self.pos, self.orientation

        self.go_to_cell((0, 0))
        self.plan.append(Actions.CLIMB)
        if self.t + len(self.plan) < self.t_max:
            self.pos, self.orientation = goal_pos, goal_orientation
            self.plan = plan
        else:
            self.pos, self.orientation = current_pos, current_orientation
            self.plan.clear()
            self.go_to_cell((0, 0))
            self.plan.append(Actions.CLIMB)
        self.t += len(self.plan)

    def get_action(self, percept: tuple[bool, bool, bool, bool, bool], reward: int) -> Actions:
        if self.last_action == Actions.SHOOT:
            self.kb.find_wumpus()
            if self.kb.wumpus is None:
                stench_point = list(self.kb.stench_frontier)[0]
                adjacent_stench_frontier = self.kb.get_adjacent_cells(stench_point, 0)
                shoot_cell = self.get_closest_adjacent_stench_cell(
                    stench_point) if self.shoot_pit is None else self.shoot_pit
                if percept[Percepts.SCREAM]:
                    self.kb.wumpus = shoot_cell
                else:
                    for cell in adjacent_stench_frontier:
                        if cell != shoot_cell:
                            x, y = cell
                            self.kb.frontier.remove(cell)
                            self.kb.cell_status[x][y] = 2
                            if len(adjacent_stench_frontier) == 2:
                                self.kb.wumpus = cell

        if len(self.plan) > 0:
            self.last_action = self.plan[0]
            return self.plan.popleft()

        if percept[Percepts.GLITTER]:
            current_pos, current_orientation = self.pos, self.orientation
            self.plan = deque([Actions.GRAB])
            self.go_to_cell((0, 0))
            self.plan.append(Actions.CLIMB)
            self.reconfigure_path(current_pos, current_orientation)
            self.last_action = self.plan[0]
            return self.plan.popleft()

        self.kb.add_percepts_to_cell(self.pos, percept)
        self.kb.add_cell_to_known(self.pos)
        self.kb.find_wumpus()

        destination: tuple[int, int] | None = self.kb.get_closest_promising_cell(self.pos, self.orientation)
        if destination is not None:
            current_pos, current_orientation = self.pos, self.orientation
            self.kb.add_cell_to_known(destination)
            self.go_to_cell(destination)
            self.reconfigure_path(current_pos, current_orientation)
            self.last_action = self.plan[0]
            return self.plan.popleft()

        if self.kb.has_arrow and self.kb.wumpus_is_not_in_pit():
            current_pos, current_orientation = self.pos, self.orientation
            self.kill_wumpus()
            self.reconfigure_path(current_pos, current_orientation)
            self.last_action = self.plan[0]
            return self.plan.popleft()

        pit: tuple[int, int] | None = self.kb.get_closest_safest_pit_cell(self.pos, self.orientation)
        if pit is not None:
            shoot: bool = self.kb.shoot_possible_pit(pit)
            current_pos, current_orientation = self.pos, self.orientation
            if not shoot or not self.kb.has_arrow:
                self.kb.add_cell_to_known(pit)
                self.go_to_cell(pit)
                self.reconfigure_path(current_pos, current_orientation)
                self.last_action = self.plan[0]
                return self.plan.popleft()

            self.shoot_pit = pit
            self.kill_wumpus(pit)
            self.reconfigure_path(current_pos, current_orientation)
            self.last_action = self.plan[0]
            return self.plan.popleft()

        # run back to start
        current_pos, current_orientation = self.pos, self.orientation
        self.go_to_cell((0, 0))
        self.plan.append(Actions.CLIMB)
        self.reconfigure_path(current_pos, current_orientation)
        self.last_action = self.plan[0]
        return self.plan.popleft()
