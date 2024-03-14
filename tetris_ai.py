import os
import sys
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
from game import Game
from tetromino import Tetromino
import random
import pickle
from common import *
from gui import Gui
import time
import multiprocessing
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models import RLModel, NewModel

# size dependent
shape_main_grid = (-1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1)
if STATE_INPUT == 'short':
    shape_hold_next = (1, 1 * 2 + 1 + 1 + 6 * Tetromino.pool_size())
    shape_hold_next_description = '[height_sum, hole_sum, combo, is_hold, 6 * 7 type] -> length = 43'
    split_hold_next = 1 * 2 + 1 + 1
else:
    shape_hold_next = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = GAME_BOARD_WIDTH * 2 + 1

shape_dense = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())

gamma = 0.95
epsilon = 0.06

current_avg_score = 0
rand = random.Random()

penalty = -500
# reward_coef = [1.0, 0.5, 0.3, 0.2]
reward_coef = [1.0, 1.0, 1.0, 1.0]
reward_coef_plan = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], 1, 50]
num_search_best = 6
num_search_rd = 6
env_debug = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filepath=None):
    if MODEL_TYPE == 'RLModel':
        model_loaded = RLModel(input_size=1).to(device)
    elif MODEL_TYPE == 'NewModel':
        model_loaded = NewModel(input_size=1).to(device)
    else:
        model_loaded = None
        sys.stderr.write('MODEL_TYPE is wrong. Exit...')
        exit()

    if filepath is not None:
        if MODEL_TYPE == 'RLModel':
            model = RLModel(input_size=1).to(device)
            model.load_state_dict(torch.load(filepath))
        elif MODEL_TYPE == 'NewModel':
            model = NewModel(input_size=1).to(device)
            model.load_state_dict(torch.load(filepath))    
    else:
        torch.save(model_loaded.state_dict(), FOLDER_NAME + 'whole_model/outer_{}'.format(0))
        print('model initial state has been saved')

    return model_loaded

def ai_play(model, max_games=100, mode='piece', is_gui_on=True):
    max_steps_per_episode = 2000
    seed = None
    gui = Gui() if is_gui_on else None
    env = Game(gui=gui, seed=seed, height=0)

    episode_count = 0
    total_score = 0

    pause_time = 0.00

    while True and episode_count < max_games:
        env.reset()
        for step in range(max_steps_per_episode):
            states, add_scores, dones, _, _, moves, _ = env.get_all_possible_states_input()
            rewards = get_reward(add_scores, dones)
            i1, i2 = split_input(states)
            q = rewards + model(i1, i2).detach().numpy()
            best = torch.argmax(q).item()
            if mode == 'step':
                best_moves = moves[best]

                for i in range(len(best_moves) - 1):
                    move = best_moves[i]
                    env.step(action=move)
                    env.render()
                    time.sleep(pause_time)
                env.step(chosen=best)
                env.render()
                time.sleep(pause_time)
            else:
                env.step(chosen=best)
                env.render()
                time.sleep(pause_time)

            if env.is_done() or step == max_steps_per_episode - 1:
                episode_count += 1
                total_score += env.current_state.score
                print('episode #{}:   score:{}'.format(episode_count, env.current_state.score))
                break

    print('average score = {:7.2f}'.format(total_score / max_games))


def ai_play_search(model, max_games=100, is_gui_on=True):
    max_steps_per_episode = 2000
    seed = None
    gui = Gui() if is_gui_on else None
    env = Game(seed=seed, height=0)
    env_gui = Game(gui=gui)

    episode_count = 0
    total_score = 0

    pause_time = 0.04

    while episode_count < max_games:
        env.reset()
        old_state = env.current_state.copy()
        moves_buffer = []
        for step in range(max_steps_per_episode):
            env_gui.current_state = old_state.copy()
            old_state = env.current_state.copy()
            moves = []
            thread = threading.Thread(target=ai_get_moves, args=(model, env, moves))
            thread.start()

            for m in moves_buffer:
                env_gui.step(action=m)
                env_gui.render()
                time.sleep(pause_time)

            thread.join()
            moves_buffer = moves

            if env.current_state.game_status == 'gameover':
                break

            if env.is_done() or step >= max_steps_per_episode - 1:
                episode_count += 1
                total_score += env.current_state.score
                print('episode #{}:   score:{}'.format(episode_count, env.current_state.score))
                break

    print('average score = {:7.2f}'.format(total_score / max_games))


def ai_get_moves(model, env, moves):
    gamestates_new, gamestates_steps, reward_prev = search_steps(model, env, num_remain=10, num_random=0, action_take=1)
    moves.clear()
    moves += env.get_moves(gamestates_steps[0][0])
    env.current_state = gamestates_steps[0][0]


def search_steps(model, env, num_remain=num_search_best, num_random=num_search_rd, action_take=1):
    gamestates_new, gamestates_steps, reward_prev = search_one_step(model, [env.current_state], env,
                                                                    num_remain=num_remain, num_random=num_random)

    save = [[], [], []]
    if action_take == 1:
        save = gamestates_new[-1], gamestates_steps[-1], reward_prev[-1]

    for _ in range(3):
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev,
                                                                        num_remain=num_remain, num_random=num_random)

    if action_take != 1:
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev, num_remain=1,
                                                                        num_random=1)
        return gamestates_new, gamestates_steps, reward_prev
    else:
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev, num_remain=1,
                                                                        num_random=0)
        gamestates_new = [gamestates_steps[0][0], save[0]]
        gamestates_steps = [gamestates_steps[0][:1], save[1]]
        reward_prev = [reward_prev[0], save[2]]
        return gamestates_new, gamestates_steps, reward_prev


def search_one_step(model, gamestates_old, env, gamestates_steps_old=None, reward_prev_old=None, num_remain=10,
                    num_random=5):
    s_all = list()
    r_all = list()
    done_all = list()
    gamestates_new = list()
    gamestates_steps_new = list()

    if gamestates_steps_old is None:
        gamestates_steps_old = [[]] * len(gamestates_old)

    if reward_prev_old is None:
        reward_prev_old = np.array([0] * len(gamestates_old))

    for i in range(len(gamestates_old)):
        states, add_scores, dones, _, _, _, gamestates = env.get_all_possible_states_input(gamestates_old[i])
        s_all.append(states)
        r_all.append(get_reward(add_scores, dones, add=reward_prev_old[i] / gamma))
        done_all += dones
        gamestates_new += gamestates
        for j in range(len(gamestates)):
            gamestates_steps_new.append(gamestates_steps_old[i].copy() + [gamestates[j]])

    s_all = np.concatenate(s_all)
    r_all = np.concatenate(r_all)
    i1, i2 = split_input(s_all)
    q = model(i1, i2).detach().numpy() + r_all
    q = torch.tensor(q)

    arg_sorted = torch.argsort(q.view(-1), descending=True).numpy().tolist()
    
    gamestates_chosen = list()
    reward_prev_chosen = list()
    gamestates_steps_chosen = list()
    prev = 0
    q_prev = -999
    num_remain = min(num_remain, len(gamestates_new))
    for _ in range(num_remain):
        if prev >= len(gamestates_new):
            break

        while q_prev == q[arg_sorted[prev]]:
            prev += 1
            if prev >= len(gamestates_new):
                break

        gamestates_chosen.append(gamestates_new[arg_sorted[prev]])
        reward_prev_chosen.append(r_all[arg_sorted[prev]])
        gamestates_steps_chosen.append(gamestates_steps_new[arg_sorted[prev]])
        q_prev = q[arg_sorted[prev]]

        if done_all[arg_sorted[prev]]:
            break

        prev += 1

    num_random = min(num_random, len(gamestates_new))
    for _ in range(num_random):
        rd_int = random.randint(0, len(gamestates_new) - 1)
        gamestates_chosen.append(gamestates_new[rd_int])
        reward_prev_chosen.append(r_all[rd_int])
        gamestates_steps_chosen.append(gamestates_steps_new[rd_int])

    return gamestates_chosen, gamestates_steps_chosen, reward_prev_chosen


def split_input(states):
    if STATE_INPUT == 'dense':
        return states
    else:
        if not torch.is_tensor(states):
            states = torch.tensor(states)
        divide = GAME_BOARD_HEIGHT * GAME_BOARD_WIDTH
        remainder = int(states.size(dim=1)) - divide
        in1, in2 = torch.split(states, [divide, remainder], dim=1)
        return in1.view((-1,1,20,10)).to(torch.float), in2.to(torch.float)


def gamestates_to_training_data(env, gamestates_steps):
    row_data = list()

    gamestate_prev = env.current_state
    for i in range(len(gamestates_steps)):
        s_ = env.get_state_input(gamestate_prev)
        sp_ = env.get_state_input(gamestates_steps[i])
        add_score_ = gamestates_steps[i].score - gamestate_prev.score
        done = gamestates_steps[i].game_status == 'gameover'
        row_data.append((s_, sp_, add_score_, done))
        gamestate_prev = gamestates_steps[i]
        if done:
            break

    return row_data


def get_data_from_playing_cnn2d(model_filename, target_size=8000, max_steps_per_episode=2000, proc_num=0,
                                queue=None):
    if MODEL_TYPE == 'RLModel':
        model = RLModel(input_size=1).to(device)
        model.load_state_dict(torch.load(model_filename))
    elif MODEL_TYPE == 'NewModel':
        model = NewModel(input_size=1).to(device)
        model.load_state_dict(torch.load(model_filename))    

    if model is None:
        print('ERROR: model has not been loaded. Check this part.')
        exit()

    global epsilon
    if proc_num == 0:
        epsilon = 0

    data = list()
    env = Game()
    episode_max = 1000
    total_score = 0
    avg_score = 0
    t_spins = 0

    for episode in range(episode_max):
        # env.reset(rand.randint(0, 10))
        env.reset()
        episode_data = list()
        for step in range(max_steps_per_episode):
            s = env.get_state_input(env.current_state)
            possible_states, add_scores, dones, is_include_hold, is_new_hold, _, _ = env.get_all_possible_states_input()
            rewards = get_reward(add_scores, dones)

            pool_size = Tetromino.pool_size()

            # get the best first before modifying the last next
            i1, i2 = split_input(possible_states)
            model.eval()
            q = rewards + model(i1, i2).detach().numpy()
            for j in range(len(dones)):
                if dones[j]:
                    q[j] = rewards[j]
            best = torch.argmax(q).item()
            model.train()

            # if hold was empty, then we don't know what's next; if hold was not empty, then we know what's next!
            if is_include_hold and not is_new_hold:
                possible_states[1][:-1, -pool_size:] = 0
            else:
                possible_states[1][:, -pool_size:] = 0

            rand_fl = rand.random()
            if rand_fl > epsilon:
                chosen = best
            else:
                # probability based on q
                # q_normal = q.reshape(-1)
                # q_normal = q_normal - np.min(q_normal) + 0.001
                # q_normal = q_normal / np.sum(q_normal) + 0.3
                # q_normal = q_normal / np.sum(q_normal)
                # chosen = np.random.choice(q_normal.shape[0], p=q_normal)

                # uniform probability
                chosen = random.randint(0, len(dones) - 1)

            episode_data.append(
                (s, (possible_states[0][best], possible_states[1][best]), add_scores[best], dones[best]))

            if add_scores[best] != int(add_scores[best]):
                t_spins += 1

            env.step(chosen=chosen)

            if env.is_done() or step == max_steps_per_episode - 1:
                data += episode_data
                total_score += env.current_state.score
                break

        if len(data) > target_size:
            print('proc_num: #{:<2d} | total episodes:{:<4d} | avg score:{:<7.2f} | data size:{} | t-spins: {}'.format(
                proc_num, episode + 1, total_score / (episode + 1), len(data), t_spins))
            avg_score = total_score / (episode + 1)
            break

    if queue is not None:
        queue.put((data, avg_score), block=False)
        return

    return data, avg_score


def get_data_from_playing_search(model_filename, target_size=8000, max_steps_per_episode=1000, proc_num=0,
                                 queue=None):
    if MODEL_TYPE == 'RLModel':
        model = RLModel(input_size=1).to(device)
        model.load_state_dict(torch.load(model_filename))
    elif MODEL_TYPE == 'NewModel':
        model = NewModel(input_size=1).to(device)
        model.load_state_dict(torch.load(model_filename)) 
        
    if model is None:
        print('ERROR: model has not been loaded. Check this part.')
        exit()

    global epsilon
    if proc_num == 0:
        epsilon = 0

    data = list()
    env = Game()
    episode_max = 1000
    total_score = 0
    avg_score = 0

    for episode in range(episode_max):
        env.reset()
        episode_data = list()
        for step in range(int(max_steps_per_episode)):
            gamestates_new, gamestates_steps, reward_prev = search_steps(model, env, action_take=5)
            episode_data += gamestates_to_training_data(env, gamestates_steps[0])

            if rand.random() > epsilon:
                env.current_state = gamestates_new[0].copy()
            else:
                env.current_state = gamestates_new[-1].copy()

            if env.is_done() or len(data) + len(episode_data) >= target_size:
                break

            if proc_num == 0:
                sys.stdout.write(
                    f'\r data: {len(data) + len(episode_data)} / {target_size} |'
                    f' score per step : {(total_score + env.current_state.score) / (len(data) + len(episode_data)):<6.2f} |'
                    f' game num : {episode + 1}')
                sys.stdout.flush()

        data += episode_data
        total_score += env.current_state.score

        if len(data) >= target_size:
            if proc_num == 0:
                print('\n proc_num: #{:<2d} | total episodes:{:<4d} | avg score:{:<7.2f} | data size:{}'.format(
                    proc_num, episode + 1, total_score / (episode + 1), len(data)))
            avg_score = total_score / (episode + 1)
            break

    if queue is not None:
        queue.put((data, avg_score), block=False)
        return

    return data, avg_score


def train(model, outer_start=0, outer_max=100):
    # outer_max: update samples
    inner_max = 5
    epoch_training = 5  # model fitting times
    batch_training = 512

    buffer_new_size = 12000
    buffer_outer_max = 4
    repeat_new_buffer = 2
    history = None

    overall_epoch_losses = []

    for outer in range(outer_start + 1, outer_start + 1 + outer_max):
        print('======== outer = {} ========'.format(outer))
        time_outer_begin = time.time()
        modify_reward_coef(outer)

        # 1. collecting data.
        buffer = list()

        # getting new samples
        new_buffer = collect_samples_multiprocess_queue(model_filename=FOLDER_NAME + f'whole_model/outer_{outer - 1}',
                                                        target_size=buffer_new_size)
        save_buffer_to_file(FOLDER_NAME + f'dataset/buffer_{outer}.pkl', new_buffer)
        buffer += new_buffer

        # load more samples. The latest dataset can be added to the buffer twice to give them larger weight.
        for i in range(max(1, outer - buffer_outer_max + 1), outer):
            buffer += load_buffer_from_file(filename=FOLDER_NAME + 'dataset/buffer_{}.pkl'.format(i))

        for _ in range(repeat_new_buffer):
            buffer += load_buffer_from_file(filename=FOLDER_NAME + 'dataset/buffer_{}.pkl'.format(outer))

        random.shuffle(buffer)

        # 2. calculating target
        s, s_, r_, dones_ = process_buffer_best(buffer)

        buffer_size = len(buffer)
        new_buffer_size = len(new_buffer)
        del buffer
        del new_buffer

        for inner in range(inner_max):
            print(f"      ======== inner = {inner + 1}/{inner_max} =========")
            target = list()
            for i in range(int(s.shape[0] / batch_training) + 1):
                start = i * batch_training
                end = min((i + 1) * batch_training, s.shape[0])
                model.eval()

                i1, i2 = split_input(s_[start:end])
                q = model(i1, i2).detach().numpy().reshape(-1) + r_[start:end]
                target.append(q)
            target = np.concatenate(target)
            # when it's gameover, Q[s'] must not be added
            for i in range(len(dones_)):
                if dones_[i]:
                    target[i] = r_[i]

            target = target * gamma
            if inner == inner_max - 1:
                save_training_dataset_to_file(filename=FOLDER_NAME + 'dataset/dataset_{}.pkl'.format(outer),
                                              dataset=(s, target))

            epoch_losses = baseline_train(model, s, target, batch_training, epoch_training)
            overall_epoch_losses += epoch_losses

        torch.save(model.state_dict(), FOLDER_NAME + 'whole_model/outer_{}'.format(outer))
        #model.save(FOLDER_NAME + 'whole_model/outer_{}'.format(outer))
        #model.save_weights(FOLDER_NAME + 'checkpoints_dqn/outer_{}'.format(outer))

        time_outer_end = time.time()
        text_ = ''
        if outer == 1:
            text_ += f'input shapes: {shape_main_grid} {shape_hold_next} \n {shape_hold_next_description} \n'

        '''text_ += 'outer = {:>4d} | pre-training avg score = {:>8.3f} | loss = {:>8.3f} | mse = {:>8.3f} |' \
                 ' dataset size = {:>7d} | new dataset size = {:>7d} | time elapsed: {:>6.1f} sec | coef = {} | penalty = {:>7d} | gamma = {:>6.3f} |' \
                 ' search best/rd = {}, {} |\n' \
            .format(outer, current_avg_score, history.history['loss'][-1], history.history['mean_squared_error'][-1],
                    buffer_size, new_buffer_size, time_outer_end - time_outer_begin, reward_coef, penalty, gamma,
                    num_search_best, num_search_rd
                    )
        append_record(text_)
        print('   ' + text_)'''

    plot_losses(overall_epoch_losses, "loss_plot")

def baseline_train(model, inputs, target, batch_size, epochs):
    criterion = nn.MSELoss()

    inputs = torch.tensor(inputs)
    target = torch.tensor(target)
    dataset = TensorDataset(inputs, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []

    for epoch_count in range(epochs):
        losses = 0
        model.train()

        for batch_input, batch_target in dataloader:
            in1, in2 = split_input(batch_input)
            logits = model(in1, in2)
            logits = logits.squeeze(1).to(torch.float)
            loss = criterion(logits, batch_target.to(torch.float))
            loss.backward()

            model.optimizer.step()
            model.zero_grad()
            losses += loss.item()

        epoch_loss = round(losses/len(dataloader),3)
        print('Epoch', (epoch_count+1), '| Losses:', epoch_loss)
        epoch_losses.append(epoch_loss)

    return epoch_losses


def save_buffer_to_file(filename, buffer):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)


def save_training_dataset_to_file(filename, dataset):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def load_buffer_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def process_buffer_best(buffer):
    s = list()
    s_ = list()
    add_scores = list()
    dones_ = list()
    for row in buffer:
        s.append(row[0])
        s_.append(row[1])
        add_scores.append(row[2])
        dones_ += [row[3]]

    s = np.concatenate(s)
    s_ = np.concatenate(s_)
    r_ = get_reward(add_scores, dones_)
    r_ = np.concatenate(r_)
    return s, s_, r_, dones_


def render_env_debug_state_input(state):
    if STATE_INPUT == 'dense':
        return

    global env_debug
    if env_debug is None:
        env_debug = Game(gui=Gui())

    loc = 0
    for r in range(GAME_BOARD_HEIGHT):
        for c in range(GAME_BOARD_WIDTH):
            env_debug.current_state.grid[r][c] = state[0, loc]
            loc += 1

    if STATE_INPUT == 'short':
        loc += 3
    else:
        loc += 21

    env_debug.render()


def render_env_debug_gamestate(gamestate):
    if STATE_INPUT == 'dense':
        return

    global env_debug
    if env_debug is None:
        env_debug = Game(gui=Gui())

    env_debug.current_state = gamestate
    env_debug.render()


def get_q_from_gamestate(model, gamestate):
    return model(split_input(Game.get_state_input(gamestate))).numpy()


def check_same_state(s1, s2):
    s1_ = s1.reshape(-1)
    s2_ = s2.reshape(-1)
    for i in range(s1_.shape[0]):
        if s1_[i] != s2_[i]: return False

    return True


def append_record(text, filename=None):
    if filename is None:
        filename = FOLDER_NAME + 'record.txt'
    with open(filename, 'a') as f:
        f.write(text)


def collect_samples_multiprocess_queue(model_filename, target_size=10000):
    timeout = 7200
    cpu_count = min(multiprocessing.cpu_count(), CPU_MAX)
    jobs = list()
    q = multiprocessing.Queue()
    for i in range(cpu_count):
        p = multiprocessing.Process(target=get_data_from_playing_search,
                                    args=(
                                        model_filename, int(target_size / cpu_count), 250, i, q))
        jobs.append(p)
        p.start()

    data = list()
    scores = list()

    for i in range(cpu_count):
        d_, s_ = q.get(timeout=timeout)
        data += d_
        scores.append(s_)

    i = 0
    for proc in jobs:
        proc.join()
        i += 1

    # average score is max(scores) because it's the process with eps = 0
    print(f'end multiprocess: total data length: {len(data)} | avg score: {max(scores):<7.2f}')
    global current_avg_score
    current_avg_score = max(scores)

    return data


def modify_reward_coef(outer):
    global reward_coef
    r_1 = reward_coef_plan[0]
    r_2 = reward_coef_plan[1]
    start = reward_coef_plan[2]
    end = reward_coef_plan[3]
    for i in range(len(reward_coef)):
        rate = (outer - start) / (end - start)
        rate = min(rate, 1)
        rate = max(rate, 0)
        reward_coef[i] = r_1[i] + (r_2[i] - r_1[i]) * rate
        reward_coef[i] = round(reward_coef[i] * 1024) / 1024
    print(f' reward_coef modified to {reward_coef}')


def get_reward(add_scores, dones, add=0):
    reward = list()
    # manipulate the reward
    for i in range(len(add_scores)):
        add_score = add_scores[i]

        # give extra reward to t-spin
        # if add_score != int(add_score):
        #     add_score = add_score * 10

        if add_score >= 90:
            add_score = add_score * reward_coef[0]
        elif add_score >= 50:
            add_score = add_score * reward_coef[1]
        elif add_score >= 20:
            add_score = add_score * reward_coef[2]
        elif add_score >= 5:
            add_score = add_score * reward_coef[3]

        if dones[i]:
            add_score += penalty
        reward.append(add_score + add)
    return np.array(reward).reshape([-1, 1])

def plot_losses(train_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")
    
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    if MODE == 'human_player':
        game = Game(gui=Gui(), seed=None)
        game.restart()
        game.run()
    elif MODE == 'ai_player_training':
        if OUT_START == 0:
            load_model()
        if MODEL_TYPE == 'RLModel':
            model_load = RLModel(input_size=1).to(device)
        elif MODEL_TYPE == 'NewModel':
            model_load = NewModel(input_size=1).to(device)

        model_load.load_state_dict(torch.load(FOLDER_NAME + 'whole_model/outer_{}'.format(OUT_START)))
        train(model_load, outer_start=OUT_START, outer_max=OUTER_MAX)
    elif MODE == 'ai_player_watching':
        if MODEL_TYPE == 'RLModel':
            model_load = RLModel(input_size=1).to(device)
        elif MODEL_TYPE == 'NewModel':
            model_load = NewModel(input_size=1).to(device)
        model_load.load_state_dict(torch.load(FOLDER_NAME + 'whole_model/outer_{}'.format(OUT_START)))
        ai_play_search(model_load, is_gui_on=True)
