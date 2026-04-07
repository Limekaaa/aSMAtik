POLICY = "RL_noLSTM"
GREEN_MAX_INVENTORY = 2
YELLOW_MAX_INVENTORY = 2
RED_MAX_INVENTORY = 1

base_save_dir = "policies/neural_networks_10_100_300_128_4_phases_CNN_hotncold_noLSTM"
model_ep = 5000
tempo_argument = {}

#kwargs = {"holding_threshold": 10,"phase": 1, "hidden_dim": 128, "deterministic": True,"green_nn_path": f"{base_save_dir}/green/green_actor_ep{model_ep}.pth", "yellow_nn_path": f"{base_save_dir}/yellow/yellow_actor_ep{model_ep}.pth", "red_nn_path": f"{base_save_dir}/red/red_actor_ep{model_ep}.pth"}
kwargs = {"holding_threshold": 10, "hidden_dim": 128, "phase":1}

green_yellow_rl_actions = [
                           {'type': 'move', 'direction': (0, 1)},
                           {'type': 'move', 'direction': (1, 0)},
                           {'type': 'move', 'direction': (0, -1)},
                           {'type': 'move', 'direction': (-1, 0)},
                           {'type': 'pick_up'},
                           {'type': 'put_down'}, 
                           {'type': 'transform'}, 
                           {'type': 'wait'},
                           {'type': 'read_message'}, 
                           {'type': 'send_message', 'recipient_ids': "g", 'content': 0},
                           {'type': 'send_message', 'recipient_ids': "g", 'content': 1},
                           {'type': 'send_message', 'recipient_ids': "g", 'content': 2}, 
                           {'type': 'send_message', 'recipient_ids': "y", 'content': 3},
                           {'type': 'send_message', 'recipient_ids': "y", 'content': 4}, 
                           {'type': 'send_message', 'recipient_ids': "y", 'content': 5}, 
                           {'type': 'send_message', 'recipient_ids': "r", 'content': 6}, 
                           {'type': 'send_message', 'recipient_ids': "r", 'content': 7}, 
                           {'type': 'send_message', 'recipient_ids': "r", 'content': 8}, 
                           {'type': 'send_message', 'recipient_ids': "gyr", 'content': 9},
                           {'type': 'send_message', 'recipient_ids': "gyr", 'content': 10},
                           {'type': 'send_message', 'recipient_ids': "gyr", 'content': 11}, 
                           {'type': 'send_message', 'recipient_ids': "gy", 'content': 12},
                           {'type': 'send_message', 'recipient_ids': "gy", 'content': 13}, 
                           {'type': 'send_message', 'recipient_ids': "gy", 'content': 14}, 
                           {'type': 'send_message', 'recipient_ids': "gr", 'content': 15},
                           {'type': 'send_message', 'recipient_ids': "gr", 'content': 16}, 
                           {'type': 'send_message', 'recipient_ids': "gr", 'content': 17}, 
                           {'type': 'send_message', 'recipient_ids': "yr", 'content': 18},
                           {'type': 'send_message', 'recipient_ids': "yr", 'content': 19}, 
                           {'type': 'send_message', 'recipient_ids': "yr", 'content': 20}]

red_rl_actions = green_yellow_rl_actions + [{'type': 'dispose'}]