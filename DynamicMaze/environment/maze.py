class Maze:
    def __init__(self, rows, cols, start, goal, walls):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        self.agent_position = start
        self.reward = 0
        self.is_done = False
        self.actions = {"East": (0, 1), "West": (0, -1), "North": (-1, 0), "South": (1, 0)}
        self.validate_maze()

    def is_inside_grid(self, position):
        if min(position) < 0 or position[0] >= self.rows or position[1] >= self.cols:
            return False
        return True

    def validate_maze(self):
        if self.start == self.goal:
            raise Exception("start is the same as goal.")
        if min(self.rows, self.cols) <= 0:
            raise Exception("Number of rows or columns must be positive.")
        for position in [self.start, self.goal]:
            if not self.is_inside_grid(position):
                raise Exception(f"{position} is out of bound.")
        for wall in self.walls:
            if not self.is_inside_grid(wall):
                raise Exception(f"{wall} is out of bound.")
        if self.start in self.walls:
            raise Exception("Invalid start.")
        if self.goal in self.walls:
            raise Exception("Invalid goal.")

    def reset(self):
        self.agent_position = self.start
        self.is_done = False
        return self.agent_position

    def step(self, action):
        if action not in self.actions:
            raise Exception("Invalid action.")
        action = self.actions[action]
        new_r, new_c = self.agent_position[0] + action[0], self.agent_position[1] + action[1]
        if not self.is_inside_grid((new_r, new_c)) or (new_r, new_c) in self.walls:
            new_r, new_c = self.agent_position
        self.agent_position = (new_r, new_c)
        if self.agent_position == self.goal:
            self.reward = 1
            self.is_done = True
        else:
            self.reward = 0
            self.is_done = False
        return self.agent_position, self.reward, self.is_done

    def change_walls(self, new_walls):
        new_walls = set(new_walls)
        for wall in new_walls:
            if not self.is_inside_grid(wall) or wall == self.start or wall == self.goal:
                raise Exception("Invalid wall")
        if self.agent_position in new_walls:
            raise Exception("Invalid wall")
        self.walls = new_walls
