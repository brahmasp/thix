from io import BytesIO
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np
from skimage.transform import resize


from embodied import Space, Env

# Based on pokegym v0.1.1, modified to be an embodied.Env environment

class Down:
    PRESS = WindowEvent.PRESS_ARROW_DOWN
    RELEASE = WindowEvent.RELEASE_ARROW_DOWN

class Left:
    PRESS = WindowEvent.PRESS_ARROW_LEFT
    RELEASE = WindowEvent.RELEASE_ARROW_LEFT

class Right:
    PRESS = WindowEvent.PRESS_ARROW_RIGHT
    RELEASE = WindowEvent.RELEASE_ARROW_RIGHT

class Up:
    PRESS = WindowEvent.PRESS_ARROW_UP
    RELEASE = WindowEvent.RELEASE_ARROW_UP

class A:
    PRESS = WindowEvent.PRESS_BUTTON_A
    RELEASE = WindowEvent.RELEASE_BUTTON_A

class B:
    PRESS = WindowEvent.PRESS_BUTTON_B
    RELEASE = WindowEvent.RELEASE_BUTTON_B

class Start:
    PRESS = WindowEvent.PRESS_BUTTON_START
    RELEASE = WindowEvent.RELEASE_BUTTON_START

ACTIONS = (Down, Left, Right, Up, A, B)

class PokemonRed(Env):
    def __init__(self, rom_path, state_path, img_size=None, headless=True, quiet=False):
        self.game, self.screen = make_env(rom_path, headless, quiet)
        self.initial_state = open_state_file(state_path)
        self.headless = headless
        self.img_size = img_size if img_size is not None else self.screen.raw_screen_buffer_dims()
        self.rescale = img_size is not None
        self.reset()

    @property
    def act_space(self):
        space = Space(np.int32, (), 0, len(ACTIONS))
        return {'action': space, 'reset': Space(bool)}

    @property
    def obs_space(self):
        return {
            'image': Space(np.uint8, self.img_size + (3,)),
            'reward': Space(np.float32),
            'is_first': Space(bool),
            'is_last': Space(bool),
            'is_terminal': Space(bool),
        }

    def reset(self):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)
        return self._obs(0.0, is_first=True)

    def render(self, high_res=False):
        pixels = self.screen.screen_ndarray()
        if not high_res and self.rescale:
            pixels = (255*resize(pixels, self.img_size + (3,))).astype(np.uint8)
        return np.copy(pixels)

    def step(self, action):
        if action['reset']:
            return self.reset()
        act_idx = action['action']
        run_action_on_emulator(self.game, self.screen, ACTIONS[act_idx], self.headless)
        return self._obs(0.0)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict(
            image=self.render(),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def close(self):
        self.game.stop()


class Pokemon(PokemonRed):

    def __init__(self, task, rom_path, state_path, img_size=None, max_steps=128, reward_scale=4.0, frame_skip=24,):
        assert task == 'red'
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.frame_skip = frame_skip
        super().__init__(rom_path, state_path, img_size)

    def reset(self):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)

        self.time = 0
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0

        self.seen_coords = set()
        self.seen_maps = set()
        self.counts_map = np.zeros((375, 500))

        self.death_count = 0
        self.total_healing = 0
        self.last_hp_fraction = 1.0
        self.last_party_size = 1
        self.last_reward = None
        self.sum_reward = 0.0

        self.max_global_x = np.iinfo(np.int16).min
        self.max_global_y = np.iinfo(np.int16).min
        self.map_progr = 0

        return self._obs(0.0, is_first=True)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False, info=None):
        obs_dict = dict(
            image=self.render(),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        if info is None:
            info = self._get_init_info()
        obs_dict.update(info)
        return obs_dict


    @property
    def obs_space(self):
        return {
            'image': Space(np.uint8, self.img_size + (3,)),
            'reward': Space(np.float32),
            'is_first': Space(bool),
            'is_last': Space(bool),
            'is_terminal': Space(bool),
            'log_obs_rew_sum': Space(np.int32),
            'log_obs_rew_delta': Space(np.float32),
            'log_obs_rew_event': Space(np.float32),
            'log_obs_rew_level': Space(np.float32),
            'log_obs_rew_opponent_level': Space(np.float32),
            'log_obs_rew_death': Space(np.float32),
            'log_obs_rew_badges': Space(np.float32),
            'log_obs_rew_healing': Space(np.float32),
            'log_obs_rew_exploration': Space(np.float32),
            'log_obs_maps_explored': Space(np.uint16),
            'log_obs_party_size':  Space(np.uint16),
            'log_obs_highest_pokemon_level':  Space(np.uint16),
            'log_obs_total_party_level':  Space(np.uint16),
            'log_obs_deaths': Space(np.uint16),
            'log_obs_badge_1': Space(np.uint16),
            'log_obs_badge_2': Space(np.uint16),
            'log_obs_event':  Space(np.uint16),
            'log_obs_money':  Space(np.uint16),
            'log_obs_max_x': Space(np.int16),
            'log_obs_max_y': Space(np.int16),
            'log_obs_map_progress_badge_1': Space(np.uint16),
            'log_obs_party': Space(np.uint8, (6,)),
        }

    def _get_init_info(self):
        info = {
            'log_obs_rew_sum': np.int32(0),
            'log_obs_rew_delta': np.float32(0.0),
            'log_obs_rew_event': np.float32(0.0),
            'log_obs_rew_level': np.float32(0.0),
            'log_obs_rew_opponent_level': np.float32(0.0),
            'log_obs_rew_death': np.float32(0.0),
            'log_obs_rew_badges': np.float32(0.0),
            'log_obs_rew_healing': np.float32(0.0),
            'log_obs_rew_exploration': np.float32(0.0),
            'log_obs_maps_explored': np.uint16(0),
            'log_obs_party_size': np.uint16(0),
            'log_obs_highest_pokemon_level': np.uint16(0),
            'log_obs_total_party_level': np.uint16(0),
            'log_obs_deaths': np.uint16(0),
            'log_obs_badge_1': np.uint16(0),
            'log_obs_badge_2': np.uint16(0),
            'log_obs_event': np.uint16(0),
            'log_obs_money': np.uint16(0),
            'log_obs_max_x': np.int16(np.iinfo(np.int16).min),
            'log_obs_max_y': np.int16(np.iinfo(np.int16).min),
            'log_obs_map_progress_badge_1': np.uint16(0),
            'log_obs_party': np.zeros(6, dtype=np.uint8),
        }
        return info

    def step(self, action):

        if action['reset']:
            return self.reset()
        act_idx = action['action']

        run_action_on_emulator(self.game, self.screen, ACTIONS[act_idx], self.headless, frame_skip=self.frame_skip)
        self.time += 1

        # Exploration reward
        x, y, map_n = ram_map_position(self.game)
        self.seen_coords.add((x, y, map_n))
        self.seen_maps.add(map_n)
        exploration_reward = 0.01 * len(self.seen_coords)
        glob_x, glob_y = game_map_local_to_global(x, y, map_n)
        try:
            self.counts_map[glob_y, glob_x] += 1
        except:
            pass

        # Log progress towards boulder badge
        if glob_x > self.max_global_x:
            self.max_global_x = glob_x
        if glob_y > self.max_global_y:
            self.max_global_y = glob_y
        try:
            map_id = ROAD_TO_BOULDER_BADGE.index(map_n)
            if map_id > self.map_progr:
                self.map_progr = map_id
        except:
            pass

        # Track current party
        party, party_size, party_levels = ram_map_party(self.game)
        party = np.array(party, dtype=np.uint8)
        party = np.select([party == 255], [0], party)  # make ambiguous 255 and 0 the same

        # Level reward
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 4 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4

        # Healing and death rewards
        hp_fraction = ram_map_hp_fraction(self.game)
        fraction_increased = hp_fraction > self.last_hp_fraction
        party_size_constant = party_size == self.last_party_size
        if fraction_increased and party_size_constant:
            if self.last_hp_fraction > 0:
                self.total_healing += hp_fraction - self.last_hp_fraction
            else:
                self.death_count += 1
        healing_reward = self.total_healing
        death_reward = 0.05 * self.death_count

        # Opponent level reward
        max_opponent_level = max(ram_map_opponent(self.game))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = 0.2 * self.max_opponent_level

        # Badge reward
        badges = ram_map_badges(self.game)
        badges_reward = 5 * badges

        # Event reward
        events = ram_map_events(self.game)
        self.max_events = max(self.max_events, events)
        event_reward = self.max_events
        money = ram_map_money(self.game)

        # Full reward
        reward = self.reward_scale * (event_reward + level_reward +
                                      opponent_level_reward + death_reward + badges_reward +
                                      healing_reward + exploration_reward)

        # Subtract previous reward
        if self.last_reward is None:
            self.last_reward = reward
            reward = 0.0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward

        self.sum_reward = self.sum_reward + self.last_reward

        done = self.time >= self.max_steps
        info = {
            'log_obs_rew_sum': np.int32(self.sum_reward),
            'log_obs_rew_delta': np.float32(reward),
            'log_obs_rew_event': np.float32(event_reward),
            'log_obs_rew_level': np.float32(level_reward),
            'log_obs_rew_opponent_level': np.float32(opponent_level_reward),
            'log_obs_rew_death': np.float32(death_reward),
            'log_obs_rew_badges': np.float32(badges_reward),
            'log_obs_rew_healing': np.float32(healing_reward),
            'log_obs_rew_exploration': np.float32(exploration_reward),
            'log_obs_maps_explored': np.uint16(len(self.seen_maps)),
            'log_obs_party_size': np.uint16(party_size),
            'log_obs_highest_pokemon_level': np.uint16(max(party_levels)),
            'log_obs_total_party_level': np.uint16(sum(party_levels)),
            'log_obs_deaths': np.uint16(self.death_count),
            'log_obs_badge_1': np.uint16(badges == 1),
            'log_obs_badge_2': np.uint16(badges > 1),
            'log_obs_event': np.uint16(events),
            'log_obs_money': np.uint16(money),
            'log_obs_max_x': np.int16(self.max_global_x),
            'log_obs_max_y': np.int16(self.max_global_y),
            'log_obs_map_progress_badge_1': np.uint16(self.map_progr),
            'log_obs_party': party.astype(np.uint8),
        }
        return self._obs(reward, is_first=False, is_last=done, info=info)


def make_env(gb_path, headless=True, quiet=False):
    game = PyBoy(
        gb_path,
        debugging=False,
        disable_input=False,
        window_type='headless' if headless else 'SDL2',
        hide_window=quiet,
    )

    screen = game.botsupport_manager().screen()

    if not headless:
        game.set_emulation_speed(6)

    return game, screen

def open_state_file(path):
    '''Load state file with BytesIO so we can cache it'''
    with open(path, 'rb') as f:
        initial_state = BytesIO(f.read())

    return initial_state

def load_pyboy_state(pyboy, state):
    '''Reset state stream and load it into PyBoy'''
    state.seek(0)
    pyboy.load_state(state)

def run_action_on_emulator(pyboy, screen, action,
        headless=True, fast_video=True, frame_skip=24):
    '''Sends actions to PyBoy'''
    press, release = action.PRESS, action.RELEASE
    pyboy.send_input(press)

    if headless or fast_video:
        pyboy._rendering(False)

    frames = []
    for i in range(frame_skip):
        if i == 8: # Release button after 8 frames
            pyboy.send_input(release)
        if not fast_video: # Save every frame
            frames.append(screen.screen_ndarray())
        if i == frame_skip - 1:
            pyboy._rendering(True)
        pyboy.tick()

    if fast_video: # Save only the last frame
        frames.append(screen.screen_ndarray())

# Memory map
# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309)
SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C
EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD761
MUSEUM_TICKET_ADDR = 0xD754
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349


def ram_map_bcd(num):
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)


def ram_map_bit_count(bits):
    return bin(bits).count('1')


def ram_map_read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit - 1] == '1'


def ram_map_read_uint16(game, start_addr):
    '''Read 2 bytes'''
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256 * val_256 + val_1


def ram_map_position(game):
    x_pos = game.get_memory_value(X_POS_ADDR)
    y_pos = game.get_memory_value(Y_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    return x_pos, y_pos, map_n


def ram_map_party(game):
    party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR]
    return party, party_size, party_levels


def ram_map_opponent(game):
    return [game.get_memory_value(addr) for addr in OPPONENT_LEVEL_ADDR]


def ram_map_oak_parcel(game):
    return ram_map_read_bit(game, OAK_PARCEL_ADDR, 1)


def ram_map_pokedex_obtained(game):
    return ram_map_read_bit(game, OAK_POKEDEX_ADDR, 5)


def ram_map_pokemon_seen(game):
    seen_bytes = [game.get_memory_value(addr) for addr in SEEN_POKE_ADDR]
    return sum([ram_map_bit_count(b) for b in seen_bytes])


def ram_map_pokemon_caught(game):
    caught_bytes = [game.get_memory_value(addr) for addr in CAUGHT_POKE_ADDR]
    return sum([ram_map_bit_count(b) for b in caught_bytes])


def ram_map_hp_fraction(game):
    party_hp = [ram_map_read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [ram_map_read_uint16(game, addr) for addr in MAX_HP_ADDR]

    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1

    return sum(party_hp) / sum_max_hp


def ram_map_money(game):
    return (100 * 100 * ram_map_bcd(game.get_memory_value(MONEY_ADDR_1))
            + 100 * ram_map_bcd(game.get_memory_value(MONEY_ADDR_100))
            + ram_map_bcd(game.get_memory_value(MONEY_ADDR_10000)))


def ram_map_badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return ram_map_bit_count(badges)


def ram_map_events(game):
    '''Adds up all event flags, exclude museum ticket'''
    num_events = sum(ram_map_bit_count(game.get_memory_value(i))
                     for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR))
    museum_ticket = int(ram_map_read_bit(game, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)

# log_obs_map_progress ie road to boulder badge
ROAD_TO_BOULDER_BADGE = [40, 0, 12, 1, 13, 50, 51, 47, 2, 54]

MAP_COORDS = {
    0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
    1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
    2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
    3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
    62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
    63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
    64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
    65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
    66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
    67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
    35: {"name": "Route 24", "coordinates": np.array([250, 235])},
    36: {"name": "Route 25", "coordinates": np.array([270, 267])},
    12: {"name": "Route 1", "coordinates": np.array([70, 43])},
    13: {"name": "Route 2", "coordinates": np.array([70, 151])},
    14: {"name": "Route 3", "coordinates": np.array([100, 179])},
    15: {"name": "Route 4", "coordinates": np.array([150, 197])},
    33: {"name": "Route 22", "coordinates": np.array([20, 71])},
    37: {"name": "Red house first", "coordinates": np.array([61, 9])},
    38: {"name": "Red house second", "coordinates": np.array([61, 0])},
    39: {"name": "Blues house", "coordinates": np.array([91, 9])},
    40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
    41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
    42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
    43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
    44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
    47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
    49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
    50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
    51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
    52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
    53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
    54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
    55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
    56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
    57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
    58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
    59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
    60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
    61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
    68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
    193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
    230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
}

POKE_IDS = {
    177: "Squirtle",
    255: "None",
    0: "None",
    165: "Rattata",
    36: "Pidgey",
    179: "Wartortle",
    150: "Pidgeotto",
    112: "Weedle",
    113: "Kakuna",
    114: "Beedrill",
    133: "Magikarp",
    169: "Geodude",
    107: "Zubat",
    109: "Paras",
    100: "Jigglypuff",
    84: "Pikachu",
    123: "Caterpie",
    124: "Metapod",
    4: "Clefairy",
    5: "Spearow",
    15: "Nidoran♀",
}

def game_map_local_to_global(x, y, map_n):
    map_x, map_y = MAP_COORDS[map_n]["coordinates"]
    return x + map_x, y + (375 - map_y)
