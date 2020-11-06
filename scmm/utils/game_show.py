'''
Simple game visualisation
'''

import matplotlib.pyplot as plt
ACTION_NO_ATTACK = 8

def game_show(color, markersize, map_size, enemies_positions, allies_positions):
    plt.clf()
    if len(enemies_positions > 0):
        plt.plot(enemies_positions[:,0], enemies_positions[:,1], color[0], markersize=20.0)
    if len(allies_positions > 0):
        plt.plot(allies_positions[:,0], allies_positions[:,1], color[1], markersize=20.0)

    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.axis('off')
            
    plt.pause(1e-10)
        
    
def game_show_adv(color, markersize, map_size, enemies, allies, actions, env):
    plt.clf()
    
    for _, e_unit in enemies.items():
        if e_unit.health > 0:
            plt.plot([e_unit.pos.x], [e_unit.pos.y], color[0], markersize=20.0, alpha=0.5)
            
    for a_id, a_unit in allies.items():
        if a_unit.health > 0:
            pos_x, pos_y = a_unit.pos.x, a_unit.pos.y
            plt.plot([pos_x], [pos_y], color[1], markersize=20.0, alpha=0.5)
            action = actions[a_id]
            if action > ACTION_NO_ATTACK:
                action = actions[a_id]
                if env.unit_to_name(a_unit) == 'medivac':
                    plt.plot([pos_x, allies[action-ACTION_NO_ATTACK].pos.x], [pos_y, allies[action-ACTION_NO_ATTACK].pos.y], 'g-.', linewidth=2)
    
                else:
                    plt.plot([pos_x, enemies[action-ACTION_NO_ATTACK].pos.x], [pos_y, enemies[action-ACTION_NO_ATTACK].pos.y], 'g-.', linewidth=2)
                
            else:
                # N S E W: 2, 3, 4, 5
                if action == 2:
                    plt.plot([pos_x, pos_x], [pos_y, pos_y+2], 'c-.', linewidth=1)
                    plt.plot([pos_x], [pos_y+2], 'gx', markersize=5)
                elif action == 3:
                    plt.plot([pos_x, pos_x], [pos_y, pos_y-1], 'c-.', linewidth=1)
                    plt.plot([pos_x], [pos_y-1], 'gx', markersize=5)
                elif action == 4:
                    plt.plot([pos_x, pos_x+1], [pos_y, pos_y], 'c-.', linewidth=1)
                    plt.plot([pos_x+1], [pos_y], 'gx', markersize=5)
                elif action == 5:
                    plt.plot([pos_x, pos_x-1], [pos_y, pos_y], 'c-.', linewidth=1)
                    plt.plot([pos_x-1], [pos_y], 'gx', markersize=5)
    

    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.axis('off')
    plt.pause(1e-10)