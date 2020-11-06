from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scmm.env.micro_env.maps import mm_maps

from pysc2 import maps as pysc2_maps


def main():
    mm_map_registry = mm_maps.get_scmm_map_registry()
    all_maps = pysc2_maps.get_maps()
    print("{:^23} {:^7} {:^7} {:^7} {:^20} {:^7}".format("Name", "Agents", "Enemies", "Limit","Type", "Symmetry"))
    for map_name, map_params in mm_map_registry.items():
        map_class = all_maps[map_name]
        if map_class.path:
            print(
                "{:^23} {:^7} {:^7} {:^7} {:^20} {:^7}".format(
                    map_name,
                    map_params["n_agents"],
                    map_params["n_enemies"],
                    map_params["limit"],
                    map_params["map_type"][0],
                    map_params["map_type"][1],
                )
            )


if __name__ == "__main__":
    main()
