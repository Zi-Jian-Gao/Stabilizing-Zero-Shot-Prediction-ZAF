import os
import sys
import argparse
import numpy as np
import yaml
import random
import agents
import task_trainers
import utils
from pathlib import Path
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

SEED = 0
REVALUATE_FLAG = False


# evaluate on all tasks seen
def evaluate_tasks(agent, result_dict, eval_args, task_list, oracle_exists, oracle_results, lb_exists, lb_results,
                   eval_ema):
    # Oracle/LB only evaluates on current task
    if agent.oracle:

        # prepare task files
        task = str(agent.task_id) + '_' + task_list[agent.task_id]['name']
        out_dir = os.path.join(agent.task_dir_dict[task], '_eval-only_' + str(agent.task_id))
        if utils.is_main_process(): Path(out_dir).mkdir(parents=True, exist_ok=True)
        eval_args['out_dir'] = out_dir
        if eval_args['lb']:
            eval_args['pretrained'] = agent.model_ckpt_history['pretrained']
        else:
            eval_args['pretrained'] = agent.model_ckpt_history[task]

        # evaluate the task
        result_file = os.path.join(out_dir, 'final_result.yaml')
        will_proceed = True
        if not REVALUATE_FLAG:
            if os.path.exists(result_file):
                try:
                    result = yaml.safe_load(open(result_file, 'r'))['result']
                    if result > 0:
                        will_proceed = False
                    else:
                        will_proceed = True
                except:
                    will_proceed = True
        if will_proceed:
            result = task_trainers.__dict__[task_list[agent.task_id]['trainer']].main(args=eval_args,
                                                                                      config=agent.task_config_dict[
                                                                                          task], eval=True,
                                                                                      test_ema=eval_ema)
            if utils.is_main_process():
                try:
                    result = float(result)
                    result = result.item()
                except:
                    pass
                yaml.dump({'result': result}, open(result_file, 'w'), default_flow_style=False)
        if utils.is_main_process(): result_dict['cl_matrix'][agent.task_id].append(result)
    # evaluate on all seen tasks
    else:
        acc_norm = []
        for t in range(agent.task_id + 1):
            # prepare task files
            task = str(t) + '_' + task_list[t]['name']
            out_dir = os.path.join(agent.task_dir_dict[task], '_eval-only_' + str(agent.task_id))
            if utils.is_main_process(): Path(out_dir).mkdir(parents=True, exist_ok=True)
            eval_args['out_dir'] = out_dir
            eval_args['pretrained'] = agent.model_ckpt_list

            # evaluate the task
            result_file = os.path.join(out_dir, 'final_result.yaml')
            will_proceed = True
            if not REVALUATE_FLAG:
                if os.path.exists(result_file):
                    try:
                        result = yaml.safe_load(open(result_file, 'r'))['result']
                        if result > 0:
                            will_proceed = False
                        else:
                            will_proceed = True
                    except:
                        will_proceed = True
            if will_proceed:

                result = task_trainers.__dict__[task_list[t]['trainer']].main(args=eval_args,
                                                                              config=agent.task_config_dict[task],
                                                                              eval=True, test_ema=eval_ema)

                # process the task results
                if utils.is_main_process():
                    try:
                        result = float(result)
                        result = result.item()
                    except:
                        pass
                    yaml.dump({'result': result}, open(result_file, 'w'), default_flow_style=False)

            if utils.is_main_process():
                result_dict['cl_matrix'][t].append(result)
                if oracle_exists:
                    if lb_exists:
                        acc_norm.append((result - lb_results[t][0]) / (oracle_results[t][0] - lb_results[t][0]))
                    else:
                        acc_norm.append(result / oracle_results[t][0])

        # post process task eval
        if utils.is_main_process():
            # compute step (final) forgetting rate
            forg_i = 0.0
            for i in range(agent.task_id ):
                to_add = max(result_dict['cl_matrix'][i]) - result_dict['cl_matrix'][i ][-1]
                forg_i += to_add
            if agent.task_id == 0:
                forg_cur = 0.0
            else:
                forg_cur = forg_i / (agent.task_id)
            result_dict['avg_forgetting'][agent.task_id] = forg_cur

    return result_dict


# train on task sequence
def trainer(args, configs, zs_config):
    # fix the seed for reproducibility
    torch.backends.cudnn.deterministic = True
    seed = SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # init world
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # create agent
    agent_config = {
        'output_dir': args.output_dir,
        'pretrained': configs['pretrained'],
        'oracle': args.oracle_flag or args.lb_flag,
        'mu': args.mu,
        'beta': args.beta,
        'text_only_flag': args.text_only_flag,
        'vision_only_flag': args.vision_only_flag,
        'global_args': args,
        'type': args.ema
    }
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)

    # get tasks
    task_list = configs['task_list']

    zsl_task_list = zs_config['task_list']

    n_tasks = len(task_list)

    # do we have an oracle? used to normalize results for averaging
    oracle_file = os.path.join(os.path.dirname(args.output_dir), 'UB/final_results/cl_matrix.yaml')
    if not os.path.exists(oracle_file):
        oracle_file = os.path.join(os.path.dirname(os.path.dirname(args.output_dir)), 'UB/final_results/cl_matrix.yaml')
    if not agent.oracle and os.path.exists(oracle_file):
        oracle_exists = True
        oracle_results = yaml.safe_load(open(oracle_file, 'r'))
    else:
        oracle_exists = False
        oracle_results = None

    # do we have a LB? used to normalize results for averaging
    lb_file = os.path.join(os.path.dirname(args.output_dir), 'LB/final_results/cl_matrix.yaml')
    if not os.path.exists(lb_file):
        lb_file = os.path.join(os.path.dirname(os.path.dirname(args.output_dir)), 'LB/final_results/cl_matrix.yaml')
    if not agent.oracle and os.path.exists(lb_file):
        lb_exists = True
        lb_results = yaml.safe_load(open(lb_file, 'r'))
    else:
        lb_exists = False
        lb_results = None

    # create results dictionary
    result_dict = {}
    if agent.oracle:
        result_keys = ['cl_matrix']
    elif oracle_exists:
        result_keys = ['cl_matrix', 'final_acc_norm', 'avg_acc_norm', 'avg_forgetting']
    else:
        result_keys = ['cl_matrix', 'avg_forgetting']
    result_dict['cl_matrix'] = [[] for t in range(n_tasks)]
    result_dict['final_acc_norm'] = [-1 for t in range(n_tasks)]
    result_dict['avg_acc_norm'] = [-1 for t in range(n_tasks)]
    result_dict['avg_forgetting'] = [-1 for t in range(n_tasks)]

    # increment over tasks
    for t in range(n_tasks):

        # get task
        task = str(t) + '_' + task_list[t]['name']
        task_config = yaml.load(open(task_list[t]['config'], 'r'), Loader=yaml.Loader)

        if args.external_lr >= 0:
            print('Overriding external LR')
            task_config['init_lr'] = args.external_lr

        with open(os.path.join(args.output_dir, 'config_task-' + task + '.yaml'), 'w') as tcf:
            yaml.dump(task_config, tcf)
        if args.debug_flag: task_config['max_epoch'] = 1
        agent.increment_task(task, task_config)

        # create task args dict
        task_args = {
            'out_dir': agent.task_dir_dict[task],
            'model_load_path': agent.model_ckpt_list,
            'model_save_path': agent.model_ckpt_history[agent.tasks[-1]],
            'device': device,
            'training_data_sample': args.training_data_sample,
            'distributed': args.distributed,
            'gpu': args.gpu,
            'pretrained': agent.model_ckpt_load,
            'agent': agent,
            'num_workers': args.num_workers,
            'eval_every': args.eval_every,
            'train_eval_type': args.train_eval_type,
            'flush_queue': args.flush_queue,
            'ema': args.ema,
            'ema_alpha': args.ema_alpha,
            'ema_lora': args.ema_lora,
            'ema_frequency': args.epoch_frequency,
            'save_frequency':args.save_frequency
        }

        # train task
        training_complete_file = os.path.join(agent.task_dir_dict[task], 'training_complete.log')
        cur_task_config = agent.task_config_dict[task]
        cur_task_config['task_seq_name'] = task_list[t]['name']
        cur_task_config['json_files'] = task_list[t].get('json_files', None)
        cur_task_config['zsl_json_files'] = zsl_task_list[0].get('json_files', None)

        cur_task_config['task_id_for_debug'] = t
        if not os.path.exists(training_complete_file) and not args.lb_flag:
            if utils.is_main_process(): print("Start training task + " + str())
            start_time = time.time()
            task_trainers.__dict__[task_list[t]['trainer']].main(args=task_args, config=cur_task_config, eval=False,
                                                                 test_ema=False)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            if utils.is_main_process(): print('Training time {}'.format(total_time_str))
            with open(training_complete_file, 'w') as f:
                f.write(total_time_str)

        # rehearsal
        if args.memory > 0:
            agent.coreset.extend(
                task_trainers.__dict__[task_list[t]['trainer']].sample_memory(memory=args.memory, args=task_args,
                                                                              config=cur_task_config, eval=False))

        # evaluate
        if not os.path.isdir(os.path.join(args.result_dir, f'task_{t:02d}')):
            eval_args = {
                'device': device,
                'training_data_sample': args.training_data_sample,
                'distributed': args.distributed,
                'gpu': args.gpu,
                'agent': agent,
                'lb': args.lb_flag,
                'num_workers': args.num_workers,
                'fast_eval': args.fast_eval,
                'flush_queue': args.flush_queue,
                'ema': args.ema,
                'ema_alpha': args.ema_alpha,
                'ema_lora': args.ema_lora,
                'ema_frequency': args.epoch_frequency,
                'save_frequency':args.save_frequency
            }

            result_dict = evaluate_tasks(agent, result_dict, eval_args, task_list, oracle_exists, oracle_results,
                                         lb_exists, lb_results, agent.ema)

            # save results
            if utils.is_main_process():
                save_dir = args.result_dir
                for rkey in result_keys:
                    with open(os.path.join(save_dir, rkey + ".yaml"), 'w') as yaml_file:
                        yaml.dump(result_dict[rkey], yaml_file, default_flow_style=False)
                save_dir = os.path.join(args.result_dir, f'task_{t:02d}')
                os.makedirs(save_dir, exist_ok=True)
                for rkey in result_keys:
                    with open(os.path.join(save_dir, rkey + ".yaml"), 'w') as yaml_file:
                        yaml.dump(result_dict[rkey], yaml_file, default_flow_style=False)
        else:
            prev_task_dir = os.path.join(args.result_dir, f'task_{t:02d}')
            for rkey in result_keys:
                with open(os.path.join(prev_task_dir, rkey + ".yaml"), 'r') as f:
                    result_dict[rkey] = yaml.safe_load(f)

        # finish task
        agent.finish_task()


def get_args():
    parser = argparse.ArgumentParser()

    # benchmark
    parser.add_argument('--config', default='./configs/continual/base.yaml')
    parser.add_argument('--output_dir', default='output/continual')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N',
                        help='Train regardless of whether saved model exists')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval_every', type=int, default=1, help="Reduce validation data evals")

    # distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local_rank", default=os.environ.get('LOCAL_RANK', 0), type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument('--debug', action='store_true',
                        help='do debug')
    parser.add_argument('--debug_port', default=12345, type=int,
                        help='for debug')
    parser.add_argument('--num_workers', default=2, type=int, help='for debug')
    parser.add_argument('--debug_addr', type=str,
                        help='for debug')
    parser.add_argument("--training_data_sample", default=1.0, type=float, help="% training data to use.")
    parser.add_argument("--memory", default=0.0, type=float, help="coreset to retain")

    # continual learning
    parser.add_argument('--agent_type', type=str, default='base', help="Base file of continual learner")
    parser.add_argument('--agent_name', type=str, default='Naive', help="Class name of continual learner")
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--lb_flag', default=False, action='store_true', help='Lower bound')
    parser.add_argument('--debug_flag', default=False, action='store_true', help='Debug mode to run faster')
    parser.add_argument('--mu', type=float, default=1.0, help="regularization strength")
    parser.add_argument('--external_lr', type=float, default=-1.0, help="regularization strength")
    parser.add_argument('--beta', type=float, default=0.0, help="regularization strength")
    parser.add_argument('--text_only_flag', default=False, action='store_true', help='only regulalarize text models')
    parser.add_argument('--vision_only_flag', default=False, action='store_true', help='only regularize vision models')
    parser.add_argument('--fast_eval', default=False, action='store_true', help='applies fast eval for multi-lora')
    parser.add_argument('--train_eval_type', type=str, default='slow',
                        help='for multi-lora training')  # slow / fast / last
    parser.add_argument('--loss_alpha', type=float, default=1.0, help="for extra losses")
    parser.add_argument('--auto_scale_alpha', default=False, action='store_true', help="for auto-scaling extra losses")
    parser.add_argument('--skip_base_keep', default=False, action='store_true',
                        help="for not keeping model -1 in adv V2")
    parser.add_argument('--force_keep', type=int, default=None, help="for adv samples CL")
    parser.add_argument('--num_adv_iters', type=int, default=11, help="for adv samples CL")
    parser.add_argument('--adv_step_sz', type=float, default=0.1, help="for adv samples CL")

    # ablations
    parser.add_argument('--adv_last_only', default=False, action='store_true', help="for adv samples CL")
    parser.add_argument('--adv_num_last', type=int, default=1, help="for adv samples CL")
    parser.add_argument('--adv_pos', default=False, action='store_true', help="for adv samples CL")

    # other
    parser.add_argument('--freeze_text_emb', default=False, action='store_true', help="for lora")
    parser.add_argument('--flush_queue', default=False, action='store_true', help='empty the queue before each task')

    # EMA setting
    parser.add_argument('--ema', type=str, default='task', help='for ema updating')  # task/epoch/mix
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='for epoch ema updating')
    parser.add_argument('--epoch_frequency', type=int, default=1, help='for epoch ema update frequency')
    parser.add_argument('--save_frequency', type=str, default='every', help='for epoch ema save') #every/best

    parser.add_argument('--ema_lora', type=str, default='continual', help='for lora initial')  # continual/zero/ema

    # zsl config
    parser.add_argument('--zsl_config', default='./configs/continual/zero_shot.yaml')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # debug
    if args.debug:
        set_remote_debugger(debug_port=args.debug_port, debug_ip=args.debug_addr)

    args.output_dir = args.output_dir.format(**args.__dict__)
    print(f'Output dir: {args.output_dir}')

    # configs, output directories, and such
    args.result_dir = os.path.join(args.output_dir, 'final_results')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    configs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    zs_config = yaml.load(open(args.zsl_config, 'r'), Loader=yaml.Loader)

    yaml.dump(configs, open(os.path.join(args.output_dir, 'config_sequence.yaml'), 'w'))
    yaml.dump(args, open(os.path.join(args.output_dir, 'args.yaml'), 'w'))

    # let's gooooooo
    trainer(args, configs, zs_config)