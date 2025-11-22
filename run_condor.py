from __future__ import print_function
from __future__ import division

import sys
import os
import argparse
import subprocess
import random
import time
import pdb
import itertools
import yaml
import numpy as np
import copy

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# saving
parser.add_argument('result_directory', default = None, help='Directory to write results to.')
parser.add_argument('--project_name', type = str, default = '')
parser.add_argument('--config', type = str, required = True)
parser.add_argument('--use_gpu', type = str2bool, default = False)
parser.add_argument('--wandb_use', type = str2bool, default = True)
parser.add_argument('--wandb_mode', type = str, default = 'offline')
parser.add_argument('--logdir', type = str, default = '~/logdir/{timestamp}/')

# common setup
parser.add_argument('--env_name', type = str, required = True)

# variables
parser.add_argument('--num_trials', default = 1, type=int, help='The number of trials to launch.')
parser.add_argument('--condor', default = False, action='store_true', help='run experiments on condor')


FLAGS = parser.parse_args()

ct = 0
EXECUTABLE = 'exp.sh'

def run_trial(outfile, cmd):

    if FLAGS.condor:
        submitFile = 'universe = container\n'
        submitFile += 'executable = ' + EXECUTABLE + "\n"
        submitFile += 'arguments = ' + cmd + '\n'
        submitFile += 'error = %s.err\n' % outfile
        submitFile += 'log = %s.log\n' % outfile
        submitFile += 'output = %s.out\n' % outfile
        # submitFile += 'error =  /dev/null\n'
        # submitFile += 'log = /dev/null\n'
        # submitFile += 'output = /dev/null\n'
        submitFile += 'should_transfer_files = YES\n'
        submitFile += 'when_to_transfer_output = ON_EXIT\n'
        submitFile += 'container_image = docker://brahmasp1/thix:v7\n'

        # squid
        setup_files = ''#'file:///staging/pavse/hrl_gpu.tar.gz'
        # staging
        common_main_files = """exp.sh, embodied, dreamerv3, utils """

        submitFile += 'transfer_input_files = {}, {}\n'.format(setup_files, common_main_files)
        submitFile += 'requirements = (OpSysMajorVer > 7)\n' # needed for apptainer
        if FLAGS.use_gpu:
            submitFile += '+WantGPULab = true\n'
            submitFile += '+GPUJobLength = "short"\n'
            submitFile += 'request_gpus = 1\n'
            submitFile += 'require_gpus = (Capability >= 8.0 && Capability < 9.0)\n'
            submitFile += 'gpus_minimum_memory = 20000\n'
        submitFile += 'request_cpus = 1\n'
        submitFile += 'request_memory = 50GB\n'
        submitFile += 'request_disk = 50GB\n'
        submitFile += 'queue'

        # with open('sample_submit.sub', 'w') as samp_sub:
        #     samp_sub.write(submitFile)
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.25)
    else:
        # TODO
        pdb.set_trace()
        #subprocess.run('"conda init bash; conda activate research; {}"'.format(cmd), shell=True)
        #cmd = 'bash -c "source activate root"' 
        subprocess.Popen(('conda run -n research ' + cmd).split())

def get_cmd(var_cfg):

    arguments = ''
    for key in var_cfg.keys():
        if key != 'algo' and key != 'env_name':
            arguments += '--{} {} '.format(key, var_cfg[key])
 
    arguments += '--configs {} {}'.format(FLAGS.env_name, var_cfg['algo'])
    print (arguments)
    if FLAGS.condor:
        cmd = '%s' % (arguments)
    return cmd

def _launch_trial(seeds, exp_name, algo, variation_name, variation_cfg):

    variation_cfg['wandb.use'] = FLAGS.wandb_use
    variation_cfg['wandb.project'] = exp_name + '-' + FLAGS.env_name
    variation_cfg['wandb.mode'] = FLAGS.wandb_mode
    variation_cfg['wandb.group'] = algo

    global ct
    for seed in seeds:
        outfile = '{}_{}_{}_{}_{}'\
            .format(FLAGS.env_name, exp_name, algo, variation_name, seed)
        if os.path.exists(outfile):
            continue
        variation_cfg['logdir'] = FLAGS.logdir + outfile
        variation_cfg['seed'] = seed
        variation_cfg['wandb.run_name'] = 'seed_{}'.format(seed)
        cmd = get_cmd(variation_cfg)
        run_trial(outfile, cmd)
        ct += 1
        print ('submitted job number: %d' % ct)

def _get_hp_combs(all_hps):
    hp_config = all_hps
    hp_names = sorted(hp_config.keys())
    all_hp_lists = []
    for hname in hp_names:
        all_hp_lists.append(hp_config[hname])
    all_combs = list(itertools.product(*all_hp_lists))
    return all_combs, hp_names

def main():  # noqa
    if FLAGS.result_directory is None:
        print('Need to provide result directory')
        sys.exit()

    #seeds = [random.randint(0, 1e6) for _ in range(FLAGS.num_trials)]
    seeds = [i for i in range(FLAGS.num_trials)]

    with open(f'dreamerv3/{FLAGS.config}', 'r') as file:
        config = yaml.safe_load(file)

    include = set([FLAGS.project_name])
    cfg_keys = []
    for exp_name in config.keys():
        if exp_name in include:
            cfg_keys.append(exp_name)

    print ('getting results for {}'.format(cfg_keys))
    for exp_name in cfg_keys:
        directory = FLAGS.result_directory + '_' + FLAGS.env_name + '_' + exp_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        common = config[exp_name]['common']
        common_hps = common['hps'] if 'hps' in common else {}
        algorithms = config[exp_name]['algorithms']
        
        for algo in algorithms:
            # get base config of algo
            algo_cfg = copy.deepcopy(algorithms[algo])
            # get algorithm specific hps
            algo_hps = copy.deepcopy(algo_cfg['hps']) if 'hps' in algo_cfg else {}
            # include common configs
            algo_cfg.update(common)
            if 'hps' in algo_cfg:
                del algo_cfg['hps'] # delete after accessing it

            # combine algo specific and common hps
            all_hps = copy.deepcopy(common_hps)
            all_hps.update(algo_hps)

            if len(all_hps):
                all_combs, hp_names = _get_hp_combs(all_hps)
                for idx, comb in enumerate(all_combs):
                    var_name = f'var{idx}'
                    for hidx, hname in enumerate(hp_names):
                        algo_cfg[hname] = comb[hidx]
                    _launch_trial(seeds, exp_name, algo, var_name, algo_cfg)
            else:
                var_name = 'var0'
                _launch_trial(seeds, exp_name, algo, var_name, algo_cfg)

    print('%d experiments ran.' % ct)

if __name__ == "__main__":
    main()

