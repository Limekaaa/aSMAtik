import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import os

# Import your custom modules
from src.model_RL import RLEnvironmentWrapper
from src.critic import GlobalCritic
import src.workspace as ws

# --- PPO Hyperparameters ---
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
UPDATE_EVERY_EPISODES = 4
MAX_EPISODES = 10001
MAX_STEPS_PER_EPISODE = 400

SAVE_INTERVAL = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Starting MAPPO Training on {device}...")
    
    # 1. Initialize Environment with baseline params to extract Master Networks
    env = RLEnvironmentWrapper(num_green_robots=1, num_yellow_robots=1, num_red_robots=1)
    env.reset()
    
    # 2. Extract Master Actor Networks
    sample_agent = env.mesa_model.robots[0]
    master_actors = {
        'green': sample_agent.policy.green_pol.to(device),
        'yellow': sample_agent.policy.yellow_pol.to(device),
        'red': sample_agent.policy.red_pol.to(device)
    }
    
    # 3. Initialize Master Critics
    critics = {
        'green': GlobalCritic(env.mesa_model , grid_width=env.mesa_model.width, grid_height=env.mesa_model.height).to(device),
        'yellow': GlobalCritic(env.mesa_model, grid_width=env.mesa_model.width, grid_height=env.mesa_model.height).to(device),
        'red': GlobalCritic(env.mesa_model, grid_width=env.mesa_model.width, grid_height=env.mesa_model.height).to(device)
    }
    
    # 4. Setup Optimizers
    actor_optimizers = {
        color: Adam(actor.parameters(), lr=LR_ACTOR) for color, actor in master_actors.items()
    }
    critic_optimizers = {
        color: Adam(critic.parameters(), lr=LR_CRITIC) for color, critic in critics.items()
    }
    
    mse_loss = nn.MSELoss()

    master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'hxs': [], 'cxs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}
    
    # --- Main Training Loop ---
    for episode in range(MAX_EPISODES):
        
        # --- RANDOMIZE ENVIRONMENT ---
        n_green = random.randint(1, 4)
        n_green = 1
        n_yellow = random.randint(1, 4)
        n_yellow = 1
        n_red = random.randint(1, 4)
        n_red = 1
        n_waste = random.randint(5, 15)
        
        env.env_kwargs['num_green_robots'] = n_green
        env.env_kwargs['num_yellow_robots'] = n_yellow
        env.env_kwargs['num_red_robots'] = n_red
        env.env_kwargs['num_initial_waste'] = n_waste
        
        env.reset()
        
        # Override Mesa's dynamically created networks with our Master Networks
        for agent in env.mesa_model.robots:
            agent.policy.green_pol = master_actors['green']
            agent.policy.yellow_pol = master_actors['yellow']
            agent.policy.red_pol = master_actors['red']

        # Buffers for trajectories, structured by agent ID
        # Added global_grids and global_coords to train the Critic
        buffers = {agent.unique_id: {
            'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 
            'values': [], 'dones': [], 'hxs': [], 'cxs': [],
            'global_grids': [], 'global_coords': [], 'masks': []
        } for agent in env.mesa_model.robots}
        
        agent_colors = {agent.unique_id: agent.robot_type for agent in env.mesa_model.robots}
        
        episode_reward = 0.0

        # --- Rollout Phase ---
        for step in range(MAX_STEPS_PER_EPISODE):
            # A. Evaluate Global State with Critics BEFORE the step
            grid_state, collector_coords = critics['green'].extract_global_state(env.mesa_model)
            grid_state = grid_state.to(device)
            collector_coords = collector_coords.to(device)
            
            with torch.no_grad():
                global_values = {
                    'green': critics['green'](grid_state, collector_coords).squeeze(),
                    'yellow': critics['yellow'](grid_state, collector_coords).squeeze(),
                    'red': critics['red'](grid_state, collector_coords).squeeze()
                }

            # B. Step the Environment (Actors make decisions here)
            with torch.no_grad():
                _, rewards, dones, _ = env.step()
            
            # Track sum of rewards for logging
            episode_reward += sum(rewards.values())

            # C. Collect Data from Agents
            for agent in env.mesa_model.robots:
                aid = agent.unique_id
                color = agent.robot_type
                k = agent.knowledge
                
                if 'last_action_idx' in k:
                    buffers[aid]['states'].append(k['last_x_input'].clone())
                    buffers[aid]['hxs'].append(k['last_hx'].clone())
                    buffers[aid]['cxs'].append(k['last_cx'].clone())
                    buffers[aid]['masks'].append(k['last_mask'].clone())
                    buffers[aid]['actions'].append(k['last_action_idx'])
                    buffers[aid]['log_probs'].append(k['last_action_log_prob'])
                    buffers[aid]['rewards'].append(rewards[aid])
                    buffers[aid]['values'].append(global_values[color].item())
                    buffers[aid]['dones'].append(dones[aid])
                    
                    # Store global states for Critic training
                    buffers[aid]['global_grids'].append(grid_state.clone())
                    buffers[aid]['global_coords'].append(collector_coords.clone())

            if dones['__all__']:
                break

        # --- GAE Phase (Per Episode) ---
        grid_state, collector_coords = critics['green'].extract_global_state(env.mesa_model)
        grid_state, collector_coords = grid_state.to(device), collector_coords.to(device)
        with torch.no_grad():
            # Fix: If episode is done, next value is mathematically 0.0
            next_values = {
                'green': 0.0 if dones['__all__'] else critics['green'](grid_state, collector_coords).item(),
                'yellow': 0.0 if dones['__all__'] else critics['yellow'](grid_state, collector_coords).item(),
                'red': 0.0 if dones['__all__'] else critics['red'](grid_state, collector_coords).item()
            }

        for color in ['green', 'yellow', 'red']:
            color_buffers = [buf for aid, buf in buffers.items() if agent_colors[aid] == color]
            
            if not color_buffers or len(color_buffers[0]['rewards']) == 0:
                continue 

            for buf in color_buffers:
                adv, ret = compute_gae(buf['rewards'], buf['values'], buf['dones'], next_values[color], GAMMA, GAE_LAMBDA)
                
                # --- NEW: Accumulate into the Master Batch ---
                master_batch[color]['states'].extend(buf['states'])
                master_batch[color]['actions'].extend(buf['actions'])
                master_batch[color]['log_probs'].extend(buf['log_probs'])
                master_batch[color]['hxs'].extend(buf['hxs'])
                master_batch[color]['cxs'].extend(buf['cxs'])
                master_batch[color]['masks'].extend(buf['masks'])
                master_batch[color]['returns'].extend(ret)
                master_batch[color]['advantages'].extend(adv)
                master_batch[color]['grids'].extend(buf['global_grids'])
                master_batch[color]['coords'].extend(buf['global_coords'])

        if episode == 0:
            avg_aloss = np.nan
            avg_closs = np.nan
            avg_ent = np.nan
        # --- PPO Update Phase (Every N Episodes) ---
        if (episode + 1) % UPDATE_EVERY_EPISODES == 0:
            ep_actor_losses, ep_critic_losses, ep_entropies = [], [], []

            for color in ['green', 'yellow', 'red']:
                mb = master_batch[color]
                if len(mb['returns']) == 0: continue
                
                # Convert the accumulated Master Batch to tensors
                states = torch.stack(mb['states']).to(device)
                hxs = torch.cat(mb['hxs']).to(device) 
                cxs = torch.cat(mb['cxs']).to(device)
                batch_masks = torch.stack(mb['masks']).to(device)
                actions = torch.tensor(mb['actions'], dtype=torch.long).to(device)
                old_log_probs = torch.tensor(mb['log_probs'], dtype=torch.float32).to(device)
                returns = torch.tensor(mb['returns'], dtype=torch.float32).to(device)
                advantages = torch.tensor(mb['advantages'], dtype=torch.float32).to(device)
                batch_grids = torch.cat(mb['grids']).to(device)
                batch_coords = torch.stack(mb['coords']).to(device)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                actor = master_actors[color]
                critic = critics[color]
                actor_optim = actor_optimizers[color]
                critic_optim = critic_optimizers[color]

                color_a_loss, color_c_loss, color_ent = 0.0, 0.0, 0.0

                for _ in range(PPO_EPOCHS):
                    # 1. Actor Update
                    logits, _, _ = actor(states, hxs, cxs)

                    ### Mask handling
                    penalty = (1.0 - batch_masks) * -1e9
                    logits = logits + penalty
                    
                    if actor.phase_1:
                        static_mask = torch.zeros_like(logits)
                        for i, action in enumerate(actor.available_actions):
                            if action['type'] in ['send_message', 'read_message']:
                                static_mask[:, i] = -1e9
                        logits = logits + static_mask
                    ###

                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    actor_optim.step()

                    # 2. Critic Update 
                    current_values = critic(batch_grids, batch_coords).squeeze()
                    if current_values.dim() == 0:
                        current_values = current_values.unsqueeze(0)
                        
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

            # --- NEW: Clear the Master Batch after the update is complete ---
            master_batch = {color: {'states': [], 'actions': [], 'log_probs': [], 'hxs': [], 'cxs': [], 'returns': [], 'advantages': [], 'grids': [], 'coords': [], 'masks': []} for color in ['green', 'yellow', 'red']}

            # Update the averages ONLY on update episodes
            avg_aloss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
            avg_closs = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
            avg_ent = np.mean(ep_entropies) if ep_entropies else 0.0


        if os.path.exists("training_log.csv"):
            with open("training_log.csv", "a") as f:
                f.write(f"{episode},{n_green},{n_yellow},{n_red},{n_waste},{env.mesa_model._count_total_waste()},{env.mesa_model._count_disposed_waste()},{episode_reward:.2f},{avg_aloss:.4f},{avg_closs:.4f},{avg_ent:.4f}\n")
        else:
            with open("training_log.csv", "w") as f:
                f.write("Episode,NumGreen,NumYellow,NumRed,NumWaste,WasteLeft,DisposedWaste,TotalReward,AvgActorLoss,AvgCriticLoss,AvgEntropy\n")
                f.write(f"{episode},{n_green},{n_yellow},{n_red},{n_waste},{env.mesa_model._count_total_waste()},{env.mesa_model._count_disposed_waste()},{episode_reward:.2f},{avg_aloss:.4f},{avg_closs:.4f},{avg_ent:.4f}\n")


        # --- Logging and Saving ---
        if episode % 5 == 0:
            
            print(f"=== Episode {episode} ===")
            print(f"Config   : {n_green} Green | {n_yellow} Yellow | {n_red} Red | {n_waste} Initial Waste")
            print(f"Progress : Waste Left: {env.mesa_model._count_total_waste()} | Disposed: {env.mesa_model._count_disposed_waste()}")
            print(f"Metrics  : Total Reward: {episode_reward:.2f}")
            print(f"Losses   : Actor: {avg_aloss:.4f} | Critic: {avg_closs:.4f} | Entropy: {avg_ent:.4f}\n")
            
        if episode % SAVE_INTERVAL == 0:
            if not os.path.exists(ws.base_save_dir):
                os.makedirs(ws.base_save_dir)
                os.makedirs(os.path.join(ws.base_save_dir, "green"), exist_ok=True)
                os.makedirs(os.path.join(ws.base_save_dir, "yellow"), exist_ok=True)
                os.makedirs(os.path.join(ws.base_save_dir, "red"), exist_ok=True)
                
            torch.save(master_actors['green'].state_dict(), os.path.join(ws.base_save_dir, "green", f"green_actor_ep{episode}.pth"))
            torch.save(master_actors['yellow'].state_dict(), os.path.join(ws.base_save_dir, "yellow", f"yellow_actor_ep{episode}.pth"))
            torch.save(master_actors['red'].state_dict(), os.path.join(ws.base_save_dir, "red", f"red_actor_ep{episode}.pth"))

if __name__ == "__main__":
    train_mappo()