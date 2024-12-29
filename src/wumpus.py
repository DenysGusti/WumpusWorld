import numpy as np
from PIL import Image
from PIL import ImageDraw
import sys
import os


class Orientation:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Actions:
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5


class Percepts:
    # [stench,breeze,glitter,bump,scream]
    STENCH = 0
    BREEZE = 1
    GLITTER = 2
    BUMP = 3
    SCREAM = 4


def get_concat_h(im1, im2):
    """
    Concatenate two images
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class Wumpus:
    def __init__(self, seed=2024, size=(4, 4), p_pit=0.2, Tmax=50):
        """
        seed ... seed for random number generator
        size ... size of the grid of the Wumpus world
        p_pit ... probability of a pit
        """
        self.rng = np.random.default_rng(seed=seed)
        self.size = size
        self.p_pit = p_pit

        # for drawing
        self.wall_width = 3  # wall width in pixels
        self.cell_size = 100  # cell size in pixels for drawing

        self.terminated = True
        self.Tmax = Tmax

        self.__load_icons()

    def __load_icons(self):
        """
        load icons
        """
        self.icons = {}

        # [stench,breeze,glitter,bump,scream]

        for name, filename in [("wumpus", "monster.png"), ("gold", "gold.png"), ("pit", "pit.png"),
                               ("agent", "robot.png"), ("north", "robot-north.png"), ("east", "robot-east.png"),
                               ("south", "robot-south.png"), ("west", "robot-west.png"), ("stench", "stench.png"),
                               ("breeze", "breeze.png"), ("glitter", "glitter.png"), ("bump", "bump.png"),
                               ("scream", "scream.png")]:
            icon = Image.open(os.path.join("icons", filename))
            icon = icon.resize((30, 30), Image.Resampling.LANCZOS)
            self.icons[name] = icon

    def __index_to_pos(self, index):
        """
        Convert index to (x,y)-coordinates
        """
        y = index // self.size[0]
        x = index - y * self.size[0]

        return x, y

    def reset(self, seed: int | None = None):
        """
        Generate a new Wumpus world
        """
        self.pos_agent = (0, 0)
        self.orientation_agent = Orientation.NORTH
        self.has_arrow = True
        self.has_gold = False
        self.terminated = False
        self.wumpus_alive = True
        self.reward = 0

        if seed is None:
            # generate pits
            pits = self.rng.random(self.size) <= self.p_pit
            pits[0, 0] = False

            # select position for wumpus
            idx_wumpus = self.rng.integers(low=1, high=np.prod(self.size))

            # select position for gold
            idx_gold = self.rng.integers(low=0, high=np.prod(self.size))
        else:
            # Ensure the seed is within the range to produce the required bits
            assert 0 <= seed < (1 << 23), "Seed must be a 23-bit integer."
            assert seed % 16 != 0, "Seed cannot have the last 4 bits as 0000 (Wumpus cannot spawn at [0, 0])."
            pits_bits = (seed >> 8) & 0x7FFF  # Extract the 15 bits for pits
            gold_bits = (seed >> 4) & 0xF  # Extract the 4 bits for gold
            wumpus_bits = seed & 0xF  # Extract the 4 bits for Wumpus

            # Generate pits
            pits = np.zeros((4, 4), dtype=bool)
            for i in range(15):  # 16 bits for a 4x4 grid excluding (0,0)
                if pits_bits & (1 << i):
                    x, y = divmod(i + 1, 4)
                    pits[x, y] = True

            # Generate Wumpus position
            idx_wumpus = wumpus_bits  # Map 4-bit value to range [1, 15]

            # Generate gold position
            idx_gold = gold_bits  # Map 4-bit value to range [0, 15]

        self.pits = np.hstack([x.reshape((-1, 1)) for x in np.where(pits)])
        self.pos_wumpus = self.__index_to_pos(idx_wumpus)
        self.pos_gold = self.__index_to_pos(idx_gold)

        # generate initial percept
        obs = obs = np.zeros((5,), dtype=int)

        stench = self.__is_stench()
        breeze = self.__is_breeze()
        glitter = self.__is_glitter()
        bump = False
        scream = False

        if stench:
            obs[Percepts.STENCH] = True
        if breeze:
            obs[Percepts.BREEZE] = True
        if glitter:
            obs[Percepts.GLITTER] = True
        if bump:
            obs[Percepts.BUMP] = True
        if scream:
            obs[Percepts.SCREAM] = True

        self.obs = obs
        self.t = 0

        return self.obs

    def __offset_for_pos(self, pos):
        """
        Get the offset for drawing for a position (top left of a cell)
        """
        x_offset = self.wall_width + pos[0] * (self.cell_size + self.wall_width)
        y_offset = self.wall_width + self.size[1] * (self.wall_width + self.cell_size) - (
                (1 + pos[1]) * (self.cell_size + self.wall_width))

        return x_offset, y_offset

    def render(self):
        """
        Render the current state of the wumpus world
        """
        height = self.size[1] * (self.cell_size + self.wall_width) + self.wall_width
        width = self.size[0] * (self.cell_size + self.wall_width) + self.wall_width
        # img = 255*np.ones((width, height, 3), dtype=int)
        img = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # grid
        sidx = 0
        for i in range(self.size[0] + 1):
            draw.rectangle((sidx, 0) + (sidx + self.wall_width, img.size[1]), fill=(0, 0, 0))
            # img[sidx:sidx+self.wall_width,:,:] = 0
            sidx += self.wall_width + self.cell_size
        sidx = 0
        for i in range(self.size[1] + 1):
            draw.rectangle((0, sidx) + (img.size[0], sidx + self.wall_width), fill=(0, 0, 0))
            # img[:,sidx:sidx+self.wall_width,:] = 0
            sidx += self.wall_width + self.cell_size

        # labels
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                x_pos = self.wall_width + 3 + x * (self.cell_size + self.wall_width)
                y_pos = img.size[1] - self.wall_width - 3 - 10 - y * (self.cell_size + self.wall_width)
                draw.text((x_pos, y_pos), f"({x + 1},{y + 1})", fill=(100, 100, 100))  # font=fnt,

        # show wumpus
        if self.wumpus_alive:
            offset = self.__offset_for_pos(self.pos_wumpus)
            img.paste(self.icons["wumpus"], (offset[0] + 10, offset[1] + 10), self.icons["wumpus"])

        # show gold
        if self.pos_gold is not None:
            offset = self.__offset_for_pos(self.pos_gold)
            img.paste(self.icons["gold"], (offset[0] + 10 + self.cell_size // 2, offset[1] + 10), self.icons["gold"])

        # show pits
        for pit in self.pits:
            pos_pit = pit
            # print("  pos pit: ", pos_pit)
            offset = self.__offset_for_pos(pos_pit)
            img.paste(self.icons["pit"], (offset[0] + 10, offset[1] + 5 + self.cell_size // 2), self.icons["pit"])

        # show robot
        offset = self.__offset_for_pos(self.pos_agent)
        if self.orientation_agent == Orientation.NORTH:
            agent_icon = self.icons["north"]
        elif self.orientation_agent == Orientation.EAST:
            agent_icon = self.icons["east"]
        elif self.orientation_agent == Orientation.SOUTH:
            agent_icon = self.icons["south"]
        elif self.orientation_agent == Orientation.WEST:
            agent_icon = self.icons["west"]
        img.paste(agent_icon, (offset[0] + 10 + self.cell_size // 2, offset[1] + 5 + self.cell_size // 2), agent_icon)

        # render percepts
        obs = self.obs
        percepts = Image.new(mode='RGB', size=(self.cell_size, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(percepts)
        draw.text((10, 10), "percepts", fill=(0, 0, 0))
        for i, k in [(Percepts.STENCH, "stench"), (Percepts.BREEZE, "breeze"), (Percepts.GLITTER, "glitter"),
                     (Percepts.BUMP, "bump"), (Percepts.SCREAM, "scream")]:
            if obs[i]:
                percepts.paste(self.icons[k], (10, 30 + i * 40), self.icons[k])

        # render status
        status = Image.new(mode='RGB', size=(20 + self.cell_size, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(status)
        draw.text((10, 10), f"reward={self.reward}", fill=(0, 0, 0))
        draw.text((10, 30), f"terminated={self.terminated}", fill=(0, 0, 0))
        draw.text((10, 50), f"arrow={self.has_arrow}", fill=(0, 0, 0))
        draw.text((10, 70), f"gold={self.has_gold}", fill=(0, 0, 0))

        return get_concat_h(get_concat_h(img, percepts), status)

    def __is_glitter(self):
        # are we on a field with gold
        if self.pos_agent == self.pos_gold:
            return True
        else:
            return False

    def __is_stench(self):
        # are we next to a wumpus?
        if self.pos_wumpus is None:
            return False
        dx = self.pos_agent[0] - self.pos_wumpus[0]
        dy = self.pos_agent[1] - self.pos_wumpus[1]
        if np.abs(dx) + np.abs(dy) <= 1:
            return True
        else:
            return False

    def __is_breeze(self):
        # are we next to a pit?
        for pos_pit in self.pits:
            dx = self.pos_agent[0] - pos_pit[0]
            dy = self.pos_agent[1] - pos_pit[1]
            if np.abs(dx) + np.abs(dy) == 1:
                return True
        return False

    def step(self, action):
        if self.terminated:
            raise AssertionError("Environment already terminated. Reset before taking any further actions.")

        # percept variables
        # [stench,breeze,glitter,bump,scream]
        stench = False
        breeze = False
        glitter = False
        bump = False
        scream = False

        reward = -1  # any actions cost
        terminated = False
        info = {}
        if action == Actions.LEFT:
            self.orientation_agent = (self.orientation_agent - 1) % 4
        elif action == Actions.RIGHT:
            self.orientation_agent = (self.orientation_agent + 1) % 4
        elif action == Actions.FORWARD:
            old_pos = self.pos_agent
            dx = 0
            dy = 0
            if self.orientation_agent == Orientation.EAST:
                dx = 1
            elif self.orientation_agent == Orientation.WEST:
                dx = -1
            elif self.orientation_agent == Orientation.NORTH:
                dy = 1
            elif self.orientation_agent == Orientation.SOUTH:
                dy = -1

            new_x = max(0, min(self.size[0] - 1, self.pos_agent[0] + dx))
            new_y = max(0, min(self.size[1] - 1, self.pos_agent[1] + dy))

            new_pos = (new_x, new_y)
            if old_pos == new_pos:
                bump = True
            self.pos_agent = new_pos
        elif action == Actions.SHOOT:
            if self.has_arrow:
                reward -= 10
                self.has_arrow = False

                # do we hit the wumpus?
                match self.orientation_agent:
                    case Orientation.NORTH:
                        dx = 0
                        dy = 1
                    case Orientation.EAST:
                        dx = 1
                        dy = 0
                    case Orientation.SOUTH:
                        dx = 0
                        dy = -1
                    case Orientation.WEST:
                        dx = -1
                        dy = 0
                for i in range(max(self.size)):
                    pos_arrow = (self.pos_agent[0] + i * dx, self.pos_agent[1] + i * dy)
                    if self.pos_wumpus == pos_arrow:
                        self.wumpus_alive = False
                        scream = True
        elif action == Actions.GRAB:
            if self.pos_agent == self.pos_gold:
                self.has_gold = True
                self.pos_gold = None
        elif action == Actions.CLIMB:
            if self.pos_agent == (0, 0):
                terminated = True
                if self.has_gold:
                    reward += 1000

        # compile percepts, check for death
        if self.wumpus_alive and (self.pos_agent == self.pos_wumpus):
            reward -= 1000
            terminated = True
            # raise ValueError("Something went wrong")
        for pos_pit in self.pits:
            pos_pit = (pos_pit[0], pos_pit[1])
            if self.pos_agent == pos_pit:
                reward = -1000
                terminated = True

        if self.pos_gold == self.pos_agent:
            glitter = True

        # compile observation vector
        #   [stench,breeze,glitter,bump,scream]
        obs = np.zeros((5,), dtype=int)
        stench = self.__is_stench()
        breeze = self.__is_breeze()
        glitter = self.__is_glitter()
        if stench:
            obs[Percepts.STENCH] = True
        if breeze:
            obs[Percepts.BREEZE] = True
        if glitter:
            obs[Percepts.GLITTER] = True
        if bump:
            obs[Percepts.BUMP] = True
        if scream:
            obs[Percepts.SCREAM] = True

        self.obs = obs

        self.t += 1
        if not terminated:
            if self.t >= (self.Tmax - 1):
                reward -= 1000
                terminated = True
                info = "Max steps reached."

        self.terminated = terminated
        self.reward = reward
        return obs, reward, terminated, info


if __name__ == "__main__":
    import pygame

    wumpus = Wumpus(1234, size=(4, 4))
    wumpus.reset()

    # initialising pygame
    pygame.init()
    img = wumpus.render()

    # creating display
    display = pygame.display.set_mode((img.width, img.height))

    img = np.array(wumpus.render()).transpose([1, 0, 2])
    surf = pygame.surfarray.make_surface(img)
    display.blit(surf, (0, 0))
    pygame.display.update()

    # creating a running loop
    pygame.event.clear()
    while True:
        # creating a loop to check events that
        # are occurring
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                wumpus.step(Actions.FORWARD)
            elif event.key == pygame.K_LEFT:
                wumpus.step(Actions.LEFT)
            elif event.key == pygame.K_RIGHT:
                wumpus.step(Actions.RIGHT)
            elif event.key == pygame.K_g:
                wumpus.step(Actions.GRAB)
            elif event.key == pygame.K_c:
                wumpus.step(Actions.CLIMB)
            elif event.key == pygame.K_t:
                wumpus.step(Actions.SHOOT)
            elif event.key == pygame.K_r:
                wumpus.reset()
            elif event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

            img = np.array(wumpus.render()).transpose([1, 0, 2])
            surf = pygame.surfarray.make_surface(img)
            display.blit(surf, (0, 0))
            pygame.display.update()
