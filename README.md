# SCMM StarCraft II Micro Management

# Installing StarCraft II
You can find lastest verson of StarCraft II [here](https://starcraft2.com/en-us/)

After installing the game, navigate to `installing_path/StarCraft II/Maps` and move a copy of all the [maps](https://github.com/caiyangcy/SCMM/tree/master/maps) in SCMM there.

If `Maps` folder does not exist, which is due to the first time of running the game, then you can manually create one

# Installing SCMM


```shell
$ git clone https://github.com/caiyangcy/SCMM.git
$ pip install SCMM/
```
You also have to install PyTorch, version 1.4.0 is one used in the project

```shell
$ pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Maps
* You can find a list of maps [here](https://github.com/caiyangcy/SC2DC/blob/master/docs/map_info.md)

Alternatively, you can run:

```shell
$ python -m scmm.bin.map_list
```

## View a Map

All the maps can be viewed by StarCraft II Editor

## Change a Map

The terrain and functionality of a map can be changed by StarCraft II Editor

## Create a Map

Create a map using StarCraft II Editor. After creation, make sure add the map to `scmm/env/micro_env/maps/mm_maps.py` and also make sure the map is added to the game folder

## Create an unit

The most important thing when creating units on a new map is to disable some reactions of them. 

To do this (taken from SMAC):

        1. Open editor, data editor, unit tab
        
        2. Right click and click add new unit
        
        3. Name the new unit, click suggest right below it
        
        4. Leave the "parent:" row alone. That determines what we're making. We want to make a unit
        
        5. Select the unit you want to copy (bottom of the new opened window, "copy from" row) e.g. zealot if you're copying zealot
        
        6. Set the "Object family:," "Race:," and "Object Type:" as desired. THESE DO NOTHING but make it easier for you to find your new unit once it's made. e.g. you probably want a new zerg unit to be in the zerg section when you go to place it on your map or something.
        
        7. Press okay, you're almost done
        
        8. Click the plus sign on the data editor tabs, go to edit actor data, actors
        
        9. Click the new actors tab
        
        10. Right click and click add new actor
        
        11. Name it and click suggest like before
        
        12. Change the "Actor Type:" row to unit
        
        13. Select what you want to copy from (bottom of the new opened window again) e.g. zealot if you're coping a zealot
        
        14. Press okay
        
        15. Click on your new actor
        
        16. At the bottom right of the window where it says "Token" and then "Unit Name," change the unit name to the name of your unit e.g. Zealot RL
        
        17. Go back to the Unit tab, find the new unit and modify the following fields:
        
            i.   (Basic) Stats: Supplies - 0
            ii.  Combat: Default Acquire Level - Passive
            iii. Behaviour: Response - No Response


## Unit Tester Map

A unit tester map can be found at unit tester map folder. Source at [unit-tester](https://www.sc2mapster.com/projects/unit-tester).

The purpose of this map is to help design some new scenarios. 


# Run

Refer to the names of agents to find out the details of running agents. 

Make sure you are under the correct folder `cd SCMM`

## Scripted

Make sure the agent name and map names match when run the scripted agents. Agents like `FocusFire`, `HybridAttackHeal`, `DyingRetreat` work on all maps but other agents like `Kiting`, `Positioning`, `BlockingEnemy` only works for some specific maps.

```shell
$ python -m scmm.agents.scripted.agent_demo --n_episodes=10 --map_name=3m --difficulty=7 --plot_level=0 --agent=FocusFire
$ python -m scmm.agents.scripted.agent_demo --n_episodes=10 --map_name=8m --difficulty=6 --plot_level=2 --agent=HybridAttackHeal --alpha=0.5
$ python -m scmm.agents.scripted.agent_demo --n_episodes=10 --map_name=3s_vs_3z_medium --difficulty=A --plot_level=0 --agent=Kiting 
```
    
## Genetic
    
```shell
$ python -m scmm.agents.genetic.ga --n_episodes=10 --map_name=8m --difficulty=7 --plot_level=0 
```
    
## NN

```shell
$ python -m scmm.agents.nn.nn --n_episodes=10 --map_name=25m --difficulty=7 --plot_level=0 
```    

## Potential Field
```shell
$ python -m scmm.agents.potential_fields.forces --n_episodes=10 --map_name=25m --difficulty=7 --plot_level=0 
```

# Plots
You can generate plots of rewards using `eval.py` under `scmm.agents`. You can find some pre-generated plots [here](https://github.com/caiyangcy/SCMM/tree/master/plots)

# Acknowledgement
* The coding on environment and part of the maps were based on [SMAC](https://github.com/oxwhirl/smac). Refer to the repo for details and license.
