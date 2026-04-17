import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import os
import pandas as pd
import torch.multiprocessing as mp

# Import your custom modules
import src.workspace as ws
from src.model_RL_RL_train import RLEnvironmentWrapper
from src.critic import GlobalCritic
import importlib
module = importlib.import_module(f'policies.{ws.POLICY}')
RLPolicies = getattr(module, 'RLPolicies')

# --- PARALLEL CONFIG ---
NUM_ENVS = 8  # Number of CPU cores to use. Reduce to 4 if you run out of RAM.

# --- PPO Hyperparameters ---
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
UPDATE_EVERY_EPISODES = 3
MAX_EPISODES = int(700001 / NUM_ENVS)
MAX_STEPS_PER_EPISODE = 400
START_EPISODE = -1
ACTOR_FREEZE_EPISODES = 0 

SAVE_INTERVAL = int(1000/NUM_ENVS)  
#phase_starts = [0, int(5000/NUM_ENVS), int(15000/NUM_ENVS), int(30000/NUM_ENVS), int(50000/NUM_ENVS)]
phase_starts = [-2000, -2000*int(5000/NUM_ENVS), -2000*int(15000/NUM_ENVS), -2000*int(30000/NUM_ENVS), -2000*int(50000/NUM_ENVS)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MULTIPROCESSING ENVIRONMENT WORKER
# ==========================================
def worker(remote, parent_remote):
    parent_remote.close()
    env = None
    while True:
        cmd, data = remote.recv()
        if cmd == 'init':
            # Intercept phase and apply it to this specific CPU's workspace memory
            if 'phase' in data:
                ws.kwargs['phase'] = data.pop('phase')
            env = RLEnvironmentWrapper(**data)
            remote.send(True)
        elif cmd == 'step':
            if env.mesa_model.is_done():
                # If this specific env is already done, return a blank waiting state
                obs = env._get_all_observations()
                dones = {a.unique_id: True for a in env.mesa_model.robots}
                dones['__all__'] = True
                rewards = {a.unique_id: 0.0 for a in env.mesa_model.robots}
                remote.send((obs, rewards, dones, {}))
            else:
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            # Intercept phase and apply it to this specific CPU's workspace memory
            if 'phase' in data:
                ws.kwargs['phase'] = data.pop('phase')
            env.env_kwargs.update(data)
            obs = env.reset()
            remote.send(obs)
        elif cmd == 'close':
            remote.close()
            break

class SubprocVecEnv:
    def __init__(self, num_envs):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [mp.Process(target=worker, args=(work_remote, remote))
                          for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def init_envs(self, kwargs_list):
        for remote, kwargs in zip(self.remotes, kwargs_list):
            remote.send(('init', kwargs))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions_list):
        for remote, action in zip(self.remotes, actions_list):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos

    def reset(self, kwargs_list):
        for remote, kwargs in zip(self.remotes, kwargs_list):
            remote.send(('reset', kwargs))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def train_mappo():
    global START_EPISODE
    print(f"Starting PARALLEL MAPPO Training ({NUM_ENVS} envs) on {device}...")
    
    # 1. Initialize Parallel Envs
    vec_envs = SubprocVecEnv(NUM_ENVS)
    dummy_kwargs = [ws.dummy_kwargs for _ in range(NUM_ENVS)]
    vec_envs.init_envs(dummy_kwargs)
    
    # 2. Extract Master Actor Networks (GPU)
    # We instantiate standalone policies since the Mesa environment is locked inside CPU processes
    dummy_env = RLEnvironmentWrapper(**dummy_kwargs[0])
    dummy_model = dummy_env.mesa_model
    
    master_actors = {
        'green': RLPolicies(dummy_model, ws.green_yellow_rl_actions, None, hidden_dim=ws.kwargs.get('hidden_dim', 64)).to(device),
        'yellow': RLPolicies(dummy_model, ws.green_yellow_rl_actions, None, hidden_dim=ws.kwargs.get('hidden_dim', 64)).to(device),
        'red': RLPolicies(dummy_model, ws.red_rl_actions, None, hidden_dim=ws.kwargs.get('hidden_dim', 64)).to(device)
    }
    
    # 3. Initialize Master Critics (GPU)
    # Using hardcoded grid sizes based on your default 20x10 layout
    critics = {
        'green': GlobalCritic(dummy_model, grid_width=20, grid_height=10).to(device),
        'yellow': GlobalCritic(dummy_model, grid_width=20, grid_height=10).to(device),
        'red': GlobalCritic(dummy_model, grid_width=20, grid_height=10).to(device)
    }

    # --- Load Checkpoints if Resuming ---
    if START_EPISODE == -1:
        if os.path.exists('training_log.csv'):
            logs = pd.read_csv('training_log.csv')
            START_EPISODE = logs['Episode'].max() if not logs.empty else 0
        else:
            START_EPISODE = 0
            
        print(f"Resuming training from Episode {START_EPISODE}...")
        for color in ['green', 'yellow', 'red']:
            actor_path = os.path.join(ws.base_save_dir, color, f"{color}_actor_last.pth")
            critic_path = os.path.join(ws.base_save_dir, color, f"{color}_critic_last.pth")
            if os.path.exists(actor_path):
                master_actors[color].load_state_dict(torch.load(actor_path, map_location=device))
                print(f"  -> Successfully loaded {color} actor from {actor_path}")
            if os.path.exists(critic_path):
                critics[color].load_state_dict(torch.load(critic_path, map_location=device))
                print(f"  -> Successfully loaded {color} critic from {critic_path}")
                
    elif START_EPISODE > 0:
        print(f"Resuming training from Episode {START_EPISODE}...")
        for color in ['green', 'yellow', 'red']:
            actor_path = os.path.join(ws.base_save_dir, color, f"{color}_actor_ep{START_EPISODE}.pth")
            critic_path = os.path.join(ws.base_save_dir, color, f"{color}_critic_ep{START_EPISODE}.pth")
            if os.path.exists(actor_path):
                master_actors[color].load_state_dict(torch.load(actor_path, map_location=device))
                print(f"  -> Successfully loaded {color} actor from {actor_path}")
            if os.path.exists(critic_path):
                critics[color].load_state_dict(torch.load(critic_path, map_location=device))
                print(f"  -> Successfully loaded {color} critic from {critic_path}")
    # -----------------------------------------
    
    # 4. Setup Optimizers
    actor_optimizers = {color: Adam(actor.parameters(), lr=LR_ACTOR) for color, actor in master_actors.items()}
    critic_optimizers = {color: Adam(critic.parameters(), lr=LR_CRITIC) for color, critic in critics.items()}
    mse_loss = nn.MSELoss()

    if 'noLSTM' in ws.POLICY:
        master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}
    else:
        master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'hxs': [], 'cxs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}
    
    # --- Main Training Loop ---
    for episode in range(START_EPISODE, MAX_EPISODES):

        # --- Phase selector ---
        base_phase = 1
        for i in range(len(phase_starts)):
            if episode >= phase_starts[i]:
                base_phase = i + 1

        curr_phase = base_phase
        if base_phase > 1:
            episodes_into_phase = episode - phase_starts[base_phase - 1]
            if episodes_into_phase < int(1500 / NUM_ENVS):
                if random.random() < 0.30:
                    curr_phase = base_phase - 1
        ws.kwargs['phase'] = curr_phase
        
        # Sync phase down to actors
        for act in master_actors.values():
            act.phase_1 = (curr_phase == 1)
        
        # --- RANDOMIZE ENVIRONMENTS ---
        # Generate independent configs for all parallel environments
        env_configs = []
        n_waste_tracker = []
        for _ in range(NUM_ENVS):
            n_waste = random.randint(5, 15)
            n_waste_tracker.append(n_waste)
            if ws.kwargs.get('rand_num_initial_green_robots', False):
                env_configs.append({
                    'num_green_robots': random.randint(1, 5), 'num_yellow_robots': 1, 'num_red_robots': 1,
                    'num_initial_waste': n_waste, 'phase': curr_phase
                })
            else:
                env_configs.append({
                    'num_green_robots': 1, 'num_yellow_robots': 1, 'num_red_robots': 1,
                    'num_initial_waste': n_waste, 'phase': curr_phase
                })
            
        obs_list = vec_envs.reset(env_configs)
        
        # Setup independent LSTM hidden states per environment and per color
        hidden_size = ws.kwargs.get('hidden_dim', 128)
        hxs = {env_idx: {c: torch.zeros(1, hidden_size).to(device) for c in ['green', 'yellow', 'red']} for env_idx in range(NUM_ENVS)}
        cxs = {env_idx: {c: torch.zeros(1, hidden_size).to(device) for c in ['green', 'yellow', 'red']} for env_idx in range(NUM_ENVS)}

        # Setup isolated buffers for each environment
        if 'noLSTM' in ws.POLICY:
            buffers = {env_idx: {color: {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'global_grids': [], 'global_coords': [], 'masks': []} for color in ['green', 'yellow', 'red']} for env_idx in range(NUM_ENVS)}
        else:
            buffers = {env_idx: {color: {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'hxs': [], 'cxs': [], 'global_grids': [], 'global_coords': [], 'masks': []} for color in ['green', 'yellow', 'red']} for env_idx in range(NUM_ENVS)}
        
        episode_rewards = [0.0] * NUM_ENVS
        dones_tracker = [False] * NUM_ENVS

        # --- Rollout Phase (Batched GPU Inference) ---
        for step in range(MAX_STEPS_PER_EPISODE):
            
            # 1. Prepare Batch for Global Critics
            batch_grids = torch.stack([obs['global_grid'] for obs in obs_list]).to(device)
            batch_coords = torch.stack([obs['collector_coords'] for obs in obs_list]).to(device)
            
            with torch.no_grad():
                global_values = {
                    'green': critics['green'](batch_grids, batch_coords).squeeze(),
                    'yellow': critics['yellow'](batch_grids, batch_coords).squeeze(),
                    'red': critics['red'](batch_grids, batch_coords).squeeze()
                }

            # 2. Prepare Batch for Actors
            actions_to_send = [{} for _ in range(NUM_ENVS)]
            step_data = {env_idx: {} for env_idx in range(NUM_ENVS)}
            
            for color in ['green', 'yellow', 'red']:
                color_states, color_masks, color_hxs, color_cxs, env_mapping = [], [], [], [], []
                
                # Gather observations across all parallel envs
                for env_idx, obs in enumerate(obs_list):
                    if dones_tracker[env_idx]: continue
                    
                    for aid, agent_data in obs.items():
                        if aid in ['global_grid', 'collector_coords']: continue
                        if agent_data['color'] == color:
                            color_states.append(agent_data['x_input'].squeeze(0))
                            color_masks.append(agent_data['mask'].squeeze(0))
                            color_hxs.append(hxs[env_idx][color])
                            color_cxs.append(cxs[env_idx][color])
                            env_mapping.append((env_idx, aid))
                
                if len(color_states) > 0:
                    batch_states = torch.stack(color_states).to(device)
                    batch_masks = torch.stack(color_masks).to(device)
                    batch_hxs = torch.cat(color_hxs, dim=0).to(device)
                    batch_cxs = torch.cat(color_cxs, dim=0).to(device)
                    
                    # SINGLE GPU FORWARD PASS FOR ALL ENVIRONMENTS
                    with torch.no_grad():
                        if not 'noLSTM' in ws.POLICY:
                            logits, new_hxs, new_cxs = master_actors[color](batch_states, batch_hxs, batch_cxs)
                        else:
                            logits = master_actors[color](batch_states)
                            
                        # Apply dynamic masks
                        penalty = (1.0 - batch_masks) * -1e9
                        logits = logits + penalty
                        
                        # Apply static comms mask for Phase 1
                        if master_actors[color].phase_1:
                            static_mask = torch.zeros_like(logits)
                            for i, action in enumerate(master_actors[color].available_actions):
                                if action['type'] in ['send_message', 'read_message']:
                                    static_mask[:, i] = -1e9
                            logits = logits + static_mask

                        dist = torch.distributions.Categorical(logits=logits)
                        action_idxs = dist.sample()
                        log_probs = dist.log_prob(action_idxs)
                        
                    # Distribute chosen actions back to their respective CPU environments
                    for i, (env_idx, aid) in enumerate(env_mapping):
                        act_idx = action_idxs[i].item()
                        
                        # Fetch the raw action dictionary
                        action_dict = master_actors[color].available_actions[act_idx].copy()
                        actions_to_send[env_idx][aid] = action_dict
                        
                        if not 'noLSTM' in ws.POLICY:
                            hxs[env_idx][color] = new_hxs[i:i+1]
                            cxs[env_idx][color] = new_cxs[i:i+1]
                            
                        # Save step data locally for GAE calculations
                        val_idx = env_idx if global_values[color].dim() > 0 else 0
                        step_data[env_idx][color] = {
                            'state': color_states[i], 'mask': color_masks[i],
                            'action': act_idx, 'log_prob': log_probs[i].item(),
                            'hx': color_hxs[i] if not 'noLSTM' in ws.POLICY else None,
                            'cx': color_cxs[i] if not 'noLSTM' in ws.POLICY else None,
                            'value': global_values[color][val_idx].item()
                        }

            # 3. Step all environments in parallel
            next_obs_list, rews_list, dones_list, _ = vec_envs.step(actions_to_send)
            
            # 4. Fill trajectory buffers
            for env_idx in range(NUM_ENVS):
                if dones_tracker[env_idx]: continue
                
                # Accumulate rewards for logging
                episode_rewards[env_idx] += sum(rews_list[env_idx].values())
                
                for color, data in step_data[env_idx].items():
                    # Find the specific agent ID corresponding to this color in this env
                    aid = next(k for k, v in obs_list[env_idx].items() if k not in ['global_grid', 'collector_coords'] and v['color'] == color)
                    
                    buffers[env_idx][color]['states'].append(data['state'])
                    buffers[env_idx][color]['masks'].append(data['mask'])
                    buffers[env_idx][color]['actions'].append(data['action'])
                    buffers[env_idx][color]['log_probs'].append(data['log_prob'])
                    buffers[env_idx][color]['values'].append(data['value'])
                    buffers[env_idx][color]['rewards'].append(rews_list[env_idx][aid])
                    buffers[env_idx][color]['dones'].append(dones_list[env_idx][aid])
                    buffers[env_idx][color]['global_grids'].append(obs_list[env_idx]['global_grid'])
                    buffers[env_idx][color]['global_coords'].append(obs_list[env_idx]['collector_coords'])
                    
                    if not 'noLSTM' in ws.POLICY:
                        buffers[env_idx][color]['hxs'].append(data['hx'])
                        buffers[env_idx][color]['cxs'].append(data['cx'])
                        
                dones_tracker[env_idx] = dones_tracker[env_idx] or dones_list[env_idx]['__all__']

            obs_list = next_obs_list
            
            # Break early only if ALL 8 environments have finished clearing the board
            if all(dones_tracker):
                break

        # --- GAE Phase (Per Environment) ---
        for env_idx in range(NUM_ENVS):
            # Extract final terminal state for this environment
            final_grid = obs_list[env_idx]['global_grid'].unsqueeze(0).to(device)
            final_coords = obs_list[env_idx]['collector_coords'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                next_values = {
                    'green': 0.0 if dones_tracker[env_idx] else critics['green'](final_grid, final_coords).item(),
                    'yellow': 0.0 if dones_tracker[env_idx] else critics['yellow'](final_grid, final_coords).item(),
                    'red': 0.0 if dones_tracker[env_idx] else critics['red'](final_grid, final_coords).item()
                }

            for color in ['green', 'yellow', 'red']:
                buf = buffers[env_idx][color]
                if len(buf['rewards']) == 0: continue 

                adv, ret = compute_gae(buf['rewards'], buf['values'], buf['dones'], next_values[color], GAMMA, GAE_LAMBDA)
                
                master_batch[color]['states'].extend(buf['states'])
                master_batch[color]['actions'].extend(buf['actions'])
                master_batch[color]['log_probs'].extend(buf['log_probs'])
                master_batch[color]['masks'].extend(buf['masks'])
                master_batch[color]['returns'].extend(ret)
                master_batch[color]['advantages'].extend(adv)
                master_batch[color]['grids'].extend(buf['global_grids'])
                master_batch[color]['coords'].extend(buf['global_coords'])
                
                if not 'noLSTM' in ws.POLICY:
                    master_batch[color]['hxs'].extend(buf['hxs'])
                    master_batch[color]['cxs'].extend(buf['cxs'])

        # --- Variables for Logging ---
        if episode == START_EPISODE:
            avg_aloss = avg_closs = avg_ent = np.nan
            
        freeze_actor = (START_EPISODE > 0) and (episode < START_EPISODE + ACTOR_FREEZE_EPISODES)
        avg_ep_reward = np.mean(episode_rewards)
        avg_waste_start = np.mean(n_waste_tracker)

        # --- PPO Update Phase (Every N Macro-Episodes) ---
        if (episode + 1) % UPDATE_EVERY_EPISODES == 0:
            ep_actor_losses, ep_critic_losses, ep_entropies = [], [], []

            for color in ['green', 'yellow', 'red']:
                mb = master_batch[color]
                if len(mb['returns']) == 0: continue
                
                states = torch.stack(mb['states']).to(device)
                batch_masks = torch.stack(mb['masks']).to(device)
                actions = torch.tensor(mb['actions'], dtype=torch.long).to(device)
                old_log_probs = torch.tensor(mb['log_probs'], dtype=torch.float32).to(device)
                returns = torch.tensor(mb['returns'], dtype=torch.float32).to(device)
                advantages = torch.tensor(mb['advantages'], dtype=torch.float32).to(device)
                batch_grids = torch.stack(mb['grids']).to(device)
                batch_coords = torch.stack(mb['coords']).to(device)
                
                if not 'noLSTM' in ws.POLICY:
                    hxs = torch.cat(mb['hxs']).to(device) 
                    cxs = torch.cat(mb['cxs']).to(device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                actor = master_actors[color]
                critic = critics[color]
                actor_optim = actor_optimizers[color]
                critic_optim = critic_optimizers[color]

                color_a_loss, color_c_loss, color_ent = 0.0, 0.0, 0.0

                for _ in range(PPO_EPOCHS):
                    # 1. Actor Update
                    if not 'noLSTM' in ws.POLICY:
                        logits, _, _ = actor(states, hxs, cxs)
                    else:
                        logits = actor(states)

                    penalty = (1.0 - batch_masks) * -1e9
                    logits = logits + penalty
                    
                    if actor.phase_1:
                        static_mask = torch.zeros_like(logits)
                        for i, action in enumerate(actor.available_actions):
                            if action['type'] in ['send_message', 'read_message']:
                                static_mask[:, i] = -1e9
                        logits = logits + static_mask

                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                    if not freeze_actor:
                        actor_optim.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                        actor_optim.step()

                    # 2. Critic Update 
                    current_values = critic(batch_grids, batch_coords).squeeze()
                    if current_values.dim() == 0: current_values = current_values.unsqueeze(0)
                        
                    critic_loss = mse_loss(current_values, returns)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    critic_optim.step()

                    color_a_loss += actor_loss.item()
                    color_c_loss += critic_loss.item()
                    color_ent += entropy.item()

                ep_actor_losses.append(color_a_loss / PPO_EPOCHS)
                ep_critic_losses.append(color_c_loss / PPO_EPOCHS)
                ep_entropies.append(color_ent / PPO_EPOCHS)

            # Clear Master Batch
            if 'noLSTM' in ws.POLICY:
                master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}
            else:
                master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'hxs': [], 'cxs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}

            avg_aloss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
            avg_closs = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
            avg_ent = np.mean(ep_entropies) if ep_entropies else 0.0

        # --- CSV Logging ---
        if os.path.exists("training_log.csv"):
            with open("training_log.csv", "a") as f:
                f.write(f"{episode},1,1,1,{avg_waste_start:.1f},0,0,{avg_ep_reward:.2f},{avg_aloss:.4f},{avg_closs:.4f},{avg_ent:.4f}\n")
        else:
            with open("training_log.csv", "w") as f:
                f.write("Episode,NumGreen,NumYellow,NumRed,NumWaste,WasteLeft,DisposedWaste,TotalReward,AvgActorLoss,AvgCriticLoss,AvgEntropy\n")
                f.write(f"{episode},1,1,1,{avg_waste_start:.1f},0,0,{avg_ep_reward:.2f},{avg_aloss:.4f},{avg_closs:.4f},{avg_ent:.4f}\n")

        # --- Console Logging and Saving ---
        if episode % 5 == 0:
            print(f"=== Macro-Episode {episode}, phase {ws.kwargs['phase']} ===")
            print(f"Config   : {NUM_ENVS} Parallel Environments | Avg Initial Waste: {avg_waste_start:.1f}")
            print(f"Metrics  : Avg Total Reward: {avg_ep_reward:.2f}")
            print(f"Losses   : Actor: {avg_aloss:.4f} | Critic: {avg_closs:.4f} | Entropy: {avg_ent:.4f}\n")

        if episode % SAVE_INTERVAL == 0:
            if not os.path.exists(ws.base_save_dir):
                os.makedirs(os.path.join(ws.base_save_dir, "green"), exist_ok=True)
                os.makedirs(os.path.join(ws.base_save_dir, "yellow"), exist_ok=True)
                os.makedirs(os.path.join(ws.base_save_dir, "red"), exist_ok=True)
                
            for color in ['green', 'yellow', 'red']:
                torch.save(master_actors[color].state_dict(), os.path.join(ws.base_save_dir, color, f"{color}_actor_ep{episode}.pth"))
                torch.save(critics[color].state_dict(), os.path.join(ws.base_save_dir, color, f"{color}_critic_ep{episode}.pth"))

        try:
            for color in ['green', 'yellow', 'red']:
                torch.save(master_actors[color].state_dict(), os.path.join(ws.base_save_dir, color, f"{color}_actor_last.pth"))
                torch.save(critics[color].state_dict(), os.path.join(ws.base_save_dir, color, f"{color}_critic_last.pth"))
        except Exception as e:
            print(f"Error saving checkpoint {episode}: {e}")
        
if __name__ == "__main__":
    mp.freeze_support() # Critical for Windows PyTorch Multiprocessing
    train_mappo()