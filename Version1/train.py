from tetris import Tetris
from agent import Agent
#from plot import plot
import cProfile
import pstats
import wandb

LR = 0.01
STATES = 6
HIDDEN_SIZES = [32,32,32]
ACTIONS = 1
MAX_MEMORY = 30000
BATCH_SIZE = 128
EPOCHS = 2

class Training_Simulation:
    def __init__(self, genome, i, generation, total_games, SLOW_DROP=True, use_wandb=False, ablate_feature=None):
        self.generation = generation
        self.i = i
        self.ablate_feature = ablate_feature
        # Calculate number of states based on ablation
        states_count = 6 if ablate_feature is None else 5
        self.tetris = Tetris(i=i, SLOW_DROP=SLOW_DROP, ablate_feature=ablate_feature)
        self.weight = genome
        self.data = [MAX_MEMORY, states_count, HIDDEN_SIZES, ACTIONS, BATCH_SIZE, LR, EPOCHS, total_games]
        self.agent = Agent(self.data)
        self.use_wandb = use_wandb

    def calculate_rewards(self, best_state):
        # Reconstruct full state from potentially ablated state
        # The state dict in game.py maintains order: total_heights, bumpiness, lines_removed, holes, y_pos, pillar
        feature_names = ['total_heights', 'bumpiness', 'lines_removed', 'holes', 'y_pos', 'pillar']
        
        # If ablation is active, one feature is missing
        if self.ablate_feature:
            # Insert default value (0) for the missing feature
            state_list = list(best_state)
            feature_names_active = [f for f in feature_names if f != self.ablate_feature]
            
            # Reconstruct full state with 0 for ablated feature
            full_state = {}
            state_idx = 0
            for fname in feature_names:
                if fname == self.ablate_feature:
                    full_state[fname] = 0  # Default value for missing feature
                else:
                    full_state[fname] = state_list[state_idx]
                    state_idx += 1
            
            total_heights = full_state['total_heights']
            bumpiness = full_state['bumpiness']
            lines_removed = full_state['lines_removed']
            holes = full_state['holes']
            y_pos = full_state['y_pos']
            pillar = full_state['pillar']
        else:
            # No ablation, unpack normally
            total_heights, bumpiness, lines_removed, holes, y_pos, pillar = best_state
        
        calc_reward = 0

        # Define when the board is "half-full"
        board_half_full = total_heights >= 110 or (total_heights >= 90 and bumpiness >= 10)

        if total_heights >= 140 or (total_heights >= 110 and bumpiness >= 12):
            hole_penalty = -2.743561101942274  # Reduced penalty when board is high
        elif total_heights >= 90 or (total_heights >= 70 and bumpiness >= 9):
            hole_penalty = -4.743561101942274
        else:
            hole_penalty = self.weight['holes']

        # Discourage Placing High When Board is Low
        if total_heights <= 40:  # Board is mostly empty
            high_placement_penalty = (10 - y_pos) * 2  # Stronger penalty
        elif total_heights <= 100:  # Board is partially filled
            high_placement_penalty = (10 - y_pos)  # Moderate penalty
        else:
            high_placement_penalty = 0  # No penalty when the board is high

        # If the piece is placed in the upper 40% of the board
        if y_pos >= 12:
            calc_reward -= high_placement_penalty

        # Game over penalty
        if self.tetris.game.finished:
            calc_reward -= self.weight['game_over']  # Severe punishment for game over

        # Base survival incentive
        calc_reward += self.weight['survival_instinct']

        # Low piece placement reward (encourage low stacking)
        if y_pos >= 9:
            calc_reward += self.weight['y_pos_reward']  # Reward for stacking high when appropriate
        else:
            calc_reward -= (10 - y_pos) * 0.2 - self.weight['y_pos_punish']  # Gradual penalty for high stacking

        # Penalty for total board height
        calc_reward += self.weight['total_height'] * total_heights  # Encourage keeping the board low

        # Line clear reward (scaled)
        calc_reward += (2 ** lines_removed) * self.weight['lines_removed']  # Scaled reward for big clears
        if lines_removed == 4:
            calc_reward += 5000

        # Penalty for holes
        calc_reward += hole_penalty * holes

        # Penalty for bumpiness (prefer smooth board)
        calc_reward += self.weight['bumpiness'] * bumpiness

        # Pillar penalty
        pillar_penalty = 0
        if holes > 0 or board_half_full:
            pillar_penalty = self.weight['pillar']
        calc_reward += pillar_penalty

        return calc_reward

    def run_simulation(self,n):
        tetris = self.tetris
        agent = self.agent
        score = lines = not_trained = 0
        tetris_clears = 0
        for game_number in range(1,n+1):
            if game_number==1000:
                return
            tetris.reset()
            done = trained = False
            old_state = tetris.game.get_state()
            episode_reward = 0
            episode_lines = 0
            episode_steps = 0
            single_clears = 0
            double_clears = 0
            triple_clears = 0
            tetris_clears_episode = 0

            while not done:
                next_states = {tuple(v): k for k, v in tetris.game.calc_all_states().items()}
                if not next_states:
                    break

                best_state = agent.get_action(next_states.keys())
                lines_cleared_this_move = best_state[2]
                lines += lines_cleared_this_move
                episode_lines += lines_cleared_this_move
                episode_steps += 1
                
                # Track individual line clear types
                if lines_cleared_this_move == 1:
                    single_clears += 1
                elif lines_cleared_this_move == 2:
                    double_clears += 1
                elif lines_cleared_this_move == 3:
                    triple_clears += 1
                elif lines_cleared_this_move == 4:
                    tetris_clears += 1
                    tetris_clears_episode += 1
                    
                best_action = next_states[best_state]

                confidence = agent.q_values[-1] if agent.q_values else 0
                tetris.update_state(best_state, confidence, agent.random, agent.epsilon)

                reward, done = tetris.play_full(best_action)

                reward += self.calculate_rewards(best_state)
                episode_reward += reward
                tetris.update_rewards(reward)

                agent.remember(old_state, best_state, reward, done)
                old_state = best_state

                if agent.total_steps % 200 == 0:
                    agent.train_long_memory()
                    trained = True

            if not trained:
                agent.train_long_memory()
                not_trained += 1
                if not_trained==5:
                    agent.update_target_network()
                    not_trained = 0
            else:
                not_trained = 0

            agent.decay_epsilon(tetris.games)

            tetris.games += 1
            score += tetris.game.score
            agent.calculate_lr(tetris.games)

            # Log to wandb
            if self.use_wandb:
                avg_loss = sum(agent.losses) / len(agent.losses) if agent.losses else 0
                avg_q_value = sum(agent.q_values) / len(agent.q_values) if agent.q_values else 0
                max_q_value = max(agent.q_values) if agent.q_values else 0
                
                wandb.log({
                    'game_number': game_number,
                    'score': tetris.game.score,
                    'single_clears': single_clears,
                    'double_clears': double_clears,
                    'triple_clears': triple_clears,
                    'tetris_clears': tetris_clears_episode,
                    'episode_steps': episode_steps,
                    'hiscore': tetris.scoreboard.hiscore,
                    'episode_reward': episode_reward,
                    'episode_lines': episode_lines,
                    'total_lines': lines,
                    'total_tetris_clears': tetris_clears,
                    'epsilon': agent.epsilon,
                    'learning_rate': agent.LR,
                    'avg_loss': avg_loss,
                    'avg_q_value': avg_q_value,
                    'max_q_value': max_q_value,
                    'total_steps': agent.total_steps,
                    'memory_size': len(agent.memory)
                })

            # print(f'LR={agent.LR:.4f} |  Epsilon={agent.epsilon:.5f} at game={game_number}')

            # if tetris.games%500==0:
            #     agent.save_model()

        return tetris.scoreboard.hiscore, lines, tetris_clears

def run_game(SLOW_DROP=True, use_wandb=True, ablate_feature=None):
    genome = {
        'game_over': 189.27613725914273,
        'survival_instinct': 8.388926084018738,
        'total_height': -0.17634932529980674,
        'lines_removed': 8.594602383216944,
        'holes': -3.743561101942274,
        'bumpiness': -6.683915232551735,
        'pillar': -11.042880500059761,
        'y_pos_reward': 207.81525814829266,
        'y_pos_punish': 117.90325502640637
    }
    n = 10000
    
    # Determine run name based on ablation
    if ablate_feature:
        run_name = f"ablation_no_{ablate_feature}_slowdrop_{SLOW_DROP}"
        print(f'\n=== Running Ablation Study ===')
        print(f'Removing feature: "{ablate_feature}"')
        feature_names = ['total_heights', 'bumpiness', 'lines_removed', 'holes', 'y_pos', 'pillar']
        active_features = [f for f in feature_names if f != ablate_feature]
        print(f'Active features: {active_features}')
        print(f'State size: 5 (removed 1 from 6)')
    else:
        run_name = f"baseline_slowdrop_{SLOW_DROP}"
        print(f'\n=== Running Baseline ===')
        print(f'Using all 6 features: [total_heights, bumpiness, lines_removed, holes, y_pos, pillar]')
        print(f'State size: 6')
    
    print(f'SLOW_DROP={SLOW_DROP}')
    print(f'Wandb tracking: {use_wandb}')
    print(f'Run name: {run_name}\n')
    
    # Initialize wandb if enabled
    if use_wandb:
        states_count = 6 if ablate_feature is None else 5
        wandb.init(
            project="tetris-ai-v1",
            config={
                "genome": genome,
                "lr": LR,
                "states": states_count,
                "ablate_feature": ablate_feature if ablate_feature else "none",
                "hidden_sizes": HIDDEN_SIZES,
                "actions": ACTIONS,
                "max_memory": MAX_MEMORY,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "total_games": n,
                "slow_drop": SLOW_DROP
            },
            name=run_name
        )
    
    t = Training_Simulation(genome, 1, False, n, SLOW_DROP, use_wandb, ablate_feature)
    t.run_simulation(n)
    
    if use_wandb:
        wandb.finish()
    
    return

import time
if __name__=='__main__':
    genome = {'game_over': 189.27613725914273, 'survival_instinct': 8.388926084018738, 'total_height': -0.17634932529980674, 'lines_removed': 8.594602383216944, 'holes': -2.743561101942274, 'bumpiness': -6.683915232551735, 'pillar': -11.042880500059761, 'y_pos_reward': 207.81525814829266, 'y_pos_punish': 117.90325502640637}
    # start_time = time.perf_counter()
    # print(Training_Simulation(genome, i=0, last_generation=False).run_simulation(100)[1]/100)
    run_game(SLOW_DROP=True, use_wandb=True)  # Set use_wandb=False to disable wandb
    # cProfile.run('run_game()', 'profile_output.prof')
    # print(f"The function took {time.perf_counter() - start_time:.6f} seconds to run.")
    # p = pstats.Stats('profile_output.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats(lambda x: x >= 1)