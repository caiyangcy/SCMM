# SCMM StarCraft II Micro Management

# Installing StarCraft II
You can find lastest verson of StarCraft II at [here](https://starcraft2.com/en-us/)

After installing the game, navigate to *installing_path/StarCraft II/Maps* and move a copy of all the maps in SCMM there.

If *Maps* folder does not exist, then manually create one

# Installing SCMM
Two ways:

1. You can directly clone the repo:

        git clone https://github.com/caiyangcy/SCMM.git
    
2. Use pip install:

        pip install SC2MM


# Maps
* You can find a list of maps [here](https://github.com/caiyangcy/SC2DC/blob/master/docs/map_info.md)

Alternatively, you can run:

    python -m scmm.bin.map_list

## View a Map

All the maps can be viewed by StarCraft II Editor

## Change a Map

The terrain and functionality of a map can be changed by StarCraft II Editor

## Create a Map

Create a map using StarCraft II Editor. After creation, make sure add the map to *scmm/env/micro_env/maps/mm_maps.py* and also make sure the map is added to the game folder

## Create an unit

The most important thing when creating units on a new map is to disable some reactions of them. 

To do this (taken from SMAC):

        Open editor, data editor, unit tab
        
        Right click and click add new unit
        
        Name the new unit, click suggest right below it
        
        Leave the "parent:" row alone. That determines what we're making. We want to make a unit
        
        Select the unit you want to copy (bottom of the new opened window, "copy from" row) e.g. zealot if you're copying zealot
        
        Set the "Object family:," "Race:," and "Object Type:" as desired. THESE DO NOTHING but make it easier for you to find your new unit once it's made. e.g. you probably want a new zerg unit to be in the zerg section when you go to place it on your map or something.
        
        Press okay, you're almost done
        
        Click the plus sign on the data editor tabs, go to edit actor data, actors
        
        Click the new actors tab
        
        Right click and click add new actor
        
        Name it and click suggest like before
        
        Change the "Actor Type:" row to unit
        
        Select what you want to copy from (bottom of the new opened window again) e.g. zealot if you're coping a zealot
        
        Press okay
        
        Click on your new actor
        
        At the bottom right of the window where it says "Token" and then "Unit Name," change the unit name to the name of your unit e.g. Zealot RL
        
        Go back to the Unit tab, find the new unit and modify the following fields:
        
            (Basic) Stats: Supplies - 0
            Combat: Default Acquire Level - Passive
            Behaviour: Response - No Response


## Unit Tester Map

A unit tester map can be found at unit tester map folder. Source at *[unit-tester](https://www.sc2mapster.com/projects/unit-tester).

The purpose of this map is to help design some new scenarios. 


# Run

Refer to the names of agents to find out the details of running agents.

## Scripted

    python -m scmm.agents.scripted.agent_demo -n_episode=10 -map_name=3m -difficulty=7 -plot_level=0 -agent=FocusFire
    
## Genetic
    
    python -m scmm.agents.genetic.ga -n_episode=10 -map_name=8m -difficulty=7 -plot_level=0 
    
## NN

    python -m scmm.agents.nn.nn -n_episode=10 -map_name=25m -difficulty=7 -plot_level=0 
    
## Potential Field

    python -m scmm.agents.potential_fields.forces -n_episode=10 -map_name=25m -difficulty=7 -plot_level=0 

# Acknowledgement
* The coding is based on [SMAC](https://github.com/oxwhirl/smac). Refer to the repo for details and license.
