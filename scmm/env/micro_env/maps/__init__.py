from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scmm.env.micro_env.maps import mm_maps


def get_map_params(map_name):
    map_param_registry = mm_maps.get_scmm_map_registry()
    return map_param_registry[map_name]
