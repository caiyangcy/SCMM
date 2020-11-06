from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scmm.env.multiagentenv import MultiAgentEnv
from scmm.env.micro_env.maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
    "patrol": 17,
    'forcefield': 1526, # target: Point
    "autoturrent": 1764
}

# Refer to https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_typeenums.h
# for detailed ID list of actions

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class MMEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=0,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        debug=False,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). The rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The extra reward for winning in an episode (default is 0).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"] # episode_limit is not going to be used here but kept as a reference
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        
        # Other
        self.game_version = game_version
        self._seed = seed
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        # Actions
        self.n_actions_no_attack = 8 # Number of non-attacking actions before it
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_content = map_params["map_content"]

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        
        # used for allies
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.ghost_id = self.raven_id = self.sentry_id = 0
        self.siege_id = 0
        
        # used for enemies
        self._e_min_unit_type = 0
        self.e_marine_id = self.e_marauder_id = self.e_medivac_id = 0
        self.e_hydralisk_id = self.e_zergling_id = self.e_baneling_id = 0
        self.e_stalker_id = self.e_colossus_id = self.e_zealot_id = 0
        self.e_ghost_id = self.e_raven_id = self.e_sentry_id = 0
        self.e_siege_id = 0
                
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.field_offset = [7, 1] # the position where force field will be placed
        self.turrent_offset = [0, 0]
        self.terrain_height = None
        self.pathing_grid = None
        self._run_config = None
        self._sc2_proc = None
        self._controller = None

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)

        _map = maps.get(self.map_name)

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(window_size=self.window_size, want_rgb=False)
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self._seed)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y
        
        self.playable_x_max = map_play_area_max.x
        self.playable_x_min = map_play_area_min.x
        self.playable_y_max = map_play_area_max.y
        self.playable_y_min = map_play_area_min.x
        
        
        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                    self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data))
                .reshape(self.map_x, self.map_y)), 1) / 255

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))
        try:
            self._obs = self._controller.observe()
            self.init_units()
            self.reward_scale_rate = self.get_reward_scaling_factor()

        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        # return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            sc_action = self.get_agent_action(a_id, action)
            if sc_action:
                sc_actions.append(sc_action)
        
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info['dead_allies'] = dead_allies
        info['dead_enemies'] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1                    

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale and not self.reward_sparse:
            reward /= self.reward_scale_rate

        return reward, terminated, info
    
    def get_reward_scaling_factor(self):
        
        max_ally_hp = sum([unit.health for _, unit in self.agents.items()])
        max_damage = sum([unit.health for _, unit in self.enemies.items()])
        
        return max_ally_hp+max_damage
        
    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
                
        elif action == 6:
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["forcefield"],
                target_world_space_pos=sc_common.Point2D(
                    x=x+self.field_offset[0], y=y+self.field_offset[1]),
                unit_tags=[tag],
                queue_command=False)
            
            if self.debug:
                logging.debug("Agent {}: Releasing Force Field at Location: {}".format(a_id, (x+self.field_offset[0], y+self.field_offset[1])))
        
        elif action == 7:
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["autoturrent"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y),
                unit_tags=[tag],
                queue_command=False)
            
            if self.debug:
                logging.debug("Agent {}: Building A Auto-Turrent at Location: {}".format(a_id, (x, y)))
    
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_content == "MMM" and self.unit_to_name(unit) == 'medivac':
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def reward_battle(self):
         """
         Reward the combat. Return 0 if sparse reward is used
         Otherwise is calculated as (remaining HP of ally units + damage done to enemies) / total HP at the beginning of the game
         
         """
         
         if self.reward_sparse:
            return 0
         
         remaining_hp = sum([unit.health for _, unit in self.agents.items()])
         damage_done = sum([unit.health_max-unit.health for _, unit in self.enemies.items()])
         
         reward = remaining_hp+damage_done
         
         return reward
         

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, unit, is_ally=True):
        """Returns the shooting range for an agent."""
        switcher = {
            self.marine_id: 6,
            self.marauder_id: 6,
            self.medivac_id: 4,  
            self.stalker_id: 4,
            self.zealot_id: 2.5, # melee, origin: 0.1, changes to 2.5 for actual game
            self.colossus_id: 7,
            self.hydralisk_id: 5,
            self.zergling_id: 11,
            self.baneling_id: 0.25, 
            self.ghost_id: 6,
            self.sentry_id: 5,
            self.e_marine_id: 6, # This one might be wrong 5 or 6???
            self.e_marauder_id: 6,
            self.e_medivac_id: 4,  
            self.e_stalker_id: 4,
            self.e_zealot_id: 2.5, # melee, origin: 0.1, changes to 2.5 for actual game
            self.e_colossus_id: 7,
            self.e_hydralisk_id: 5,
            self.e_zergling_id: 11,
            self.e_baneling_id: 0.25,
            self.e_ghost_id: 6,
            self.e_sentry_id: 5
        }

        return switcher.get(unit.unit_type, 5)
    

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 8,
            self.baneling_id: 1,
            self.e_marine_id: 15,
            self.e_marauder_id: 25,
            self.e_medivac_id: 200,  # max energy
            self.e_stalker_id: 35,
            self.e_zealot_id: 22,
            self.e_colossus_id: 24,
            self.e_hydralisk_id: 10,
            self.e_zergling_id: 8,
            self.e_baneling_id: 1
        }
        return switcher.get(unit.unit_type, 15)
    
    
    def unit_damage(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 6,
            self.marauder_id: 10,
            self.stalker_id: 13,
            self.zealot_id: 8, # 8x2
            self.colossus_id: 10, # 10x2
            self.hydralisk_id: 12,
            self.zergling_id: 5,
            self.baneling_id: 16, # splash
            self.sentry_id: 6,
            self.e_marine_id: 6,
            self.e_marauder_id: 10,
            self.e_stalker_id: 13,
            self.e_zealot_id: 8, # 8x2
            self.e_colossus_id: 10, # 10x2
            self.e_hydralisk_id: 12,
            self.e_zergling_id: 5,
            self.e_baneling_id: 16, # splash
            self.e_sentry_id: 6
        }
        return switcher.get(unit.unit_type, 6)
    
    def unit_to_name(self, unit):
        """Returns the name of a unit."""
        switcher = {
            self.marine_id: "marine",
            self.marauder_id: "maurauder",
            self.stalker_id: "stalker",
            self.zealot_id: "zealot", 
            self.colossus_id: "colossus", 
            self.hydralisk_id: "hydralisk",
            self.zergling_id: "zergling",
            self.baneling_id: "baneling", 
            self.sentry_id: "sentry", 
            self.medivac_id: "medivac", 
            self.siege_id: "siege", 
            self.e_marine_id: "marine",
            self.e_marauder_id: "maurauder",
            self.e_stalker_id: "stalker",
            self.e_zealot_id: "zealot", 
            self.e_colossus_id: "colossus", 
            self.e_hydralisk_id: "hydralisk",
            self.e_zergling_id: "zergling",
            self.e_baneling_id: "baneling" ,
            self.e_sentry_id: "sentry",
            self.e_medivac_id: "medivac",
            self.e_siege_id: "siege", 
        }
        
        return switcher.get(unit.unit_type, "Unknown")

    def unit_weapon_cooldown(self, unit):
        return unit.weapon_cooldown

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        logging.info("Replay saved at: %s" % replay_path)


    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)
        if self.check_bounds(x, y) and self.pathing_grid[y, x]:
            return True
        
        return False
    
    def get_ally_positions(self, alive_only=True):
        pos = []
        for agent_id in range(self.n_agents):
            unit = self.get_unit_by_id(agent_id) 
            if unit.health > 0:
                pos.append([unit.pos.x, unit.pos.y])
        
        return np.array(pos)
                
    def get_enemy_positions(self, alive_only=True):
        pos = []
        for e_id in range(self.n_enemies):
            unit = self.get_enermy_by_id(e_id) 
            if unit.health > 0:
                pos.append([unit.pos.x, unit.pos.y])

        return np.array(pos)
    
    def get_ally_center(self, exclude_med=False):
        center_x, center_y = 0, 0
        count = 0
        for agent_id in range(self.n_agents):
            unit = self.get_unit_by_id(agent_id) 
            if unit.health > 0:
                if exclude_med and self.unit_to_name(unit) == 'medivac':
                    continue
                center_x += unit.pos.x 
                center_y += unit.pos.y 
                count += 1
            
        center_x /= count
        center_y /= count
        
        return center_x, center_y 
    
    def get_enemy_center(self):
        center_x, center_y = 0, 0
        count = 0
        target_items = self.enemies.items()
        for t_id, t_unit in target_items: # t_id starts from 0
            if t_unit.health > 0:
                center_x += t_unit.pos.x 
                center_y += t_unit.pos.y 
                count += 1
            
        center_x /= count
        center_y /= count
        
        return center_x, center_y 
        
    def count_alive_units(self, kind='ally'):
        count = 0
        if kind == 'ally':
            for agent_id in range(self.n_agents):
                unit = self.get_unit_by_id(agent_id) 
                count += unit.health > 0
        elif kind == 'enemy':
            target_items = self.enemies.items()
            for t_id, t_unit in target_items: # t_id starts from 0
                count += t_unit.health > 0
        else:
            raise NotImplementedError()
            
        return count
    
    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return (0 <= x < self.map_x and 0 <= y < self.map_y)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type
        else:  # use default SC2 unit types
            if self.map_content == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_content == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_content == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_content == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def init_units(self):
        """Initialise the units."""
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )
                    
            
            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max
            
            if self._episode_count == 0:
                min_unit_type = min(unit.unit_type for unit in self.agents.values())
                self._init_unit_types(min_unit_type)
                
                e_min_unit_type = min(unit.unit_type for unit in self.enemies.values())
                self._init_unit_types(e_min_unit_type, False)

            all_agents_created = (len(self.agents) == self.n_agents)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1

            if not updated:  # dead
                e_unit.health = 0
        
        if (n_ally_alive == 0 and n_enemy_alive > 0 or self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0 or self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _init_unit_types(self, min_unit_type, is_ally=True):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        if is_ally:
            self._min_unit_type = min_unit_type
            if self.map_content == "marines":
                self.marine_id = min_unit_type
            elif self.map_content == "stalkers_and_zealots":
                self.stalker_id = min_unit_type
                self.zealot_id = min_unit_type + 1
            elif self.map_content == "colossi_stalkers_zealots":
                self.colossus_id = min_unit_type
                self.stalker_id = min_unit_type + 1
                self.zealot_id = min_unit_type + 2
            elif self.map_content == "MMM":
                self.marauder_id = min_unit_type
                self.marine_id = min_unit_type + 1
                self.medivac_id = min_unit_type + 2
            elif self.map_content == "zealots":
                self.zealot_id = min_unit_type
            elif self.map_content == "hydralisks":
                self.hydralisk_id = min_unit_type
            elif self.map_content == "stalkers":
                self.stalker_id = min_unit_type
                self.zealot_id = min_unit_type + 1
            elif self.map_content == "colossus":
                self.colossus_id = min_unit_type
            elif self.map_content == "bane":
                self.baneling_id = min_unit_type
                self.zergling_id = min_unit_type + 1
            elif self.map_content == "marines_and_ghosts": 
                self.marine_id = min_unit_type
                self.ghost_id = min_unit_type + 1
            elif self.map_content == "sentry_and_stalkers": 
                self.sentry_id = min_unit_type
                self.stalker_id = min_unit_type + 1
            elif self.map_content == 'siege_and_marines':
                self.marine_id = min_unit_type
                self.siege_id = min_unit_type + 1
        else:
            self._e_min_unit_type = min_unit_type
            if self.map_content == "marines":
                self.e_marine_id = min_unit_type
            elif self.map_content == "stalkers_and_zealots":
                self.e_stalker_id = min_unit_type
                self.e_zealot_id = min_unit_type + 1
            elif self.map_content == "colossi_stalkers_zealots":
                self.e_colossus_id = min_unit_type
                self.e_stalker_id = min_unit_type + 1
                self.e_zealot_id = min_unit_type + 2
            elif self.map_content == "MMM":
                self.e_marauder_id = min_unit_type
                self.e_marine_id = min_unit_type + 1
                self.e_medivac_id = min_unit_type + 2
            elif self.map_content == "zealots":
                self.e_zealot_id = min_unit_type
            elif self.map_content == "hydralisks":
                self.e_hydralisk_id = min_unit_type
            elif self.map_content == "stalkers":
                self.e_zealot_id = min_unit_type
            elif self.map_content == "colossus":
                self.e_colossus_id = min_unit_type
            elif self.map_content == "bane":
                self.e_baneling_id = min_unit_type
                self.e_zergling_id = min_unit_type + 1
            elif self.map_content == "marines_and_ghosts":
                self.e_ghost_id = min_unit_type
            elif self.map_content == "sentry_and_stalkers": 
                self.e_sentry_id = min_unit_type
                self.e_stalker_id = min_unit_type + 1
            elif self.map_content == 'siege_and_marines':
                self.e_marine_id = min_unit_type
                self.e_siege_id = min_unit_type + 1
                
    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_content != "MMM":
            return False

        if ally:
            units_alive = [a for a in self.agents.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [a for a in self.enemies.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]
    
    def get_enermy_by_id(self, e_id):
        """Get unit by ID."""
        return self.enemies[e_id]
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions


    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats
    
    def get_non_attack_action_set(self):
        actions = {
            "No-Op": 0,
            "Stop":1,
            "North":2,
            "South":3,
            "East":4,
            "West":5,
            "Force Field":6
        }
        return actions