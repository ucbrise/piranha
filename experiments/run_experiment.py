
import argparse
import boto3
import glob
import json
import os
import pandas as pd
import re
import subprocess
import time
from tqdm import tqdm

# -- matplotlib stuff --

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Sans Serif"
from matplotlib.ticker import ScalarFormatter
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

DEFAULT_COLORS = ["#0077b9", "#ff7700", "#00a400", "#d62728", "#d62728", "#9467bd", '#CC79A7', '#000000']
FADED_COLORS = ["#3e9acf", "#ff7700", "#4ebf4e", "#d45b5c"]
CB_COLORS = ["#3D4F7D", "#E48C2A", "#CD4F38", "#742C14", "#EAD890"]
FADED_CB_COLORS = ["#8793B9", "#F1B270", "#E59689", "#BF8374", "#F3DD9C"]
DEFAULT_LINESTYLES = ['-', '--', ':', '-.', '-', '--', ':', '-.']
DEFAULT_MARKERS = ['o', 's', 'p', 'd', '*', 'P', '.', '2']

default_cycler = (cycler(color=DEFAULT_COLORS) + cycler(linestyle=DEFAULT_LINESTYLES) + cycler(marker=DEFAULT_MARKERS))

# -----


piranha_aws = boto3.session.Session(profile_name='piranha')
mpspdz_aws = boto3.session.Session(profile_name='mpspdz')

MODEL_PATH = 'files/models/'


def get_running_ips(client, name):

    ips = []

    instances = client.describe_instances(
        Filters=[
            {
                'Name':'tag:Name',
                'Values': [name]
            }
        ]
    )
    if len(instances['Reservations']) > 0:
        instances = instances['Reservations'][0]['Instances']
    else:
        return ips

    for inst in instances:
        if inst['State']['Name'] == 'running':
            ips.append(inst['PublicIpAddress'])

    return ips


def get_instance_ids(client, name, state):

    ids = []

    instances = client.describe_instances(
        Filters=[
            {
                'Name':'tag:Name',
                'Values': [name]
            }
        ]
    )

    instances = instances['Reservations'][0]['Instances']

    for inst in instances:
        if inst['State']['Name'] == state:
            ids.append(inst['InstanceId'])

    return ids


# NOTE: AWS actions require a valid credential file in ~/.aws/credentials with [piranha] and [mpspdz] profiles
def start_machines():

    # GPU machines - 4 in us-west and 2 in us-east
    piranha_east = piranha_aws.client('ec2', region_name='us-east-1')
    piranha_west = piranha_aws.client('ec2', region_name='us-west-2')

    for (client, name, n_machines) in [(piranha_east, 'east', 4), (piranha_west, 'west', 2)]:

        print('starting machines in..', name, end='')

        stopped_instances = get_instance_ids(client, 'jlw', 'stopped')

        if stopped_instances:
            client.start_instances(InstanceIds=stopped_instances)

        while len(get_running_ips(client, 'jlw')) != n_machines:
            print('.', end='', flush=True)
            time.sleep(3) # wait 3 seconds for the machines to come up

        print()

    # MP-SPDZ compute-optimized machines - 4 in us-west
    mpspdz_client = mpspdz_aws.client('ec2', region_name='us-west-2')

    stopped_instances = get_instance_ids(mpspdz_client, 'piranha-mpspdz', 'stopped')

    if stopped_instances:
        mpspdz_client.start_instances(InstanceIds=stopped_instances)

    while len(get_running_ips(mpspdz_client, 'piranha-mpspdz')) != 4:
        print('.', end='', flush=True)
        time.sleep(3) # wait 3 seconds for the machines to come up

    print('\nMachines running:')
    print('\tus-east (GPUs):', get_running_ips(piranha_east, 'jlw'))
    print('\tus-west (GPUs):', get_running_ips(piranha_west, 'jlw'))
    print('\tus-west (MP-SPDZ):', get_running_ips(mpspdz_client, 'piranha-mpspdz'))
    print()


# NOTE: AWS actions require a valid credential file in ~/.aws/credentials with [piranha] and [mpspdz] profiles
def suspend_machines():

    # Suspend all machines
    piranha_east = piranha_aws.client('ec2', region_name='us-east-1')
    piranha_west = piranha_aws.client('ec2', region_name='us-west-2')
    mpspdz_client = mpspdz_aws.client('ec2', region_name='us-west-2')

    for (client, name) in [(piranha_east, 'jlw'), (piranha_west, 'jlw'), (mpspdz_client, 'piranha-mpspdz')]:

        running_instances = get_instance_ids(client, name, 'running')

        client.stop_instances(InstanceIds=running_instances)

        while len(get_running_ips(client, name)) != 0:
            print('.', end='', flush=True)
            time.sleep(3) # wait 3 seconds for the machines to stop

    print('\nMachines stopped.\n')


def update_hosts():

    # Update ansible hosts.yml so we can talk to them based on their current IPs
    piranha_east = piranha_aws.client('ec2', region_name='us-east-1')
    piranha_west = piranha_aws.client('ec2', region_name='us-west-2')
    mpspdz_client = mpspdz_aws.client('ec2', region_name='us-west-2')

    lan_ips = get_running_ips(piranha_east, 'jlw')
    wan_ips = lan_ips[0:2] + get_running_ips(piranha_west, 'jlw')
    mpspdz_ips = get_running_ips(mpspdz_client, 'piranha-mpspdz')

    with open('runfiles/hosts.yml', 'w+') as f:
        new_hosts = '''
---

lan:
  hosts:
    piranha1-lan:
      ansible_host: {}
    piranha2-lan:
      ansible_host: {}
    piranha3-lan:
      ansible_host: {}
    piranha4-lan:
      ansible_host: {}
  vars:
    ansible_user: ubuntu
wan:
  hosts:
    piranha1-wan:
      ansible_host: {}
    piranha2-wan:
      ansible_host: {}
    piranha3-wan:
      ansible_host: {}
    piranha4-wan:
      ansible_host: {}
  vars:
    ansible_user: ubuntu
mpspdz:
  hosts:
    mpspdz1:
      ansible_host: {}
    mpspdz2:
      ansible_host: {}
    mpspdz3:
      ansible_host: {}
    mpspdz4:
      ansible_host: {}
  vars:
    ansible_user: ubuntu
        '''.format(*lan_ips, *wan_ips, *mpspdz_ips)

        f.writelines(new_hosts)

    return lan_ips, wan_ips, mpspdz_ips


def rebuild_piranha(target_hosts, protocol, fp, verbose):
    
    cmd = 'ansible-playbook --private-key ~/.ssh/piranha-ae -i runfiles/hosts.yml -l {} runfiles/rebuild_piranha.yml -e "float_precision={}" -e "protocol={}"'.format(
        target_hosts, 
        fp,
        protocol
    )

    if not verbose:
        cmd += ' >/dev/null 2>&1'

    status = os.system(cmd)
    if os.waitstatus_to_exitcode(status) != 0:
        print('Piranha rebuild failed with code {}'.format(os.waitstatus_to_exitcode(status)))
        exit(os.waitstatus_to_exitcode(status))


def run_piranha_experiment(target_hosts, ips, protocol, nparties, fp, configuration, network, figure, verbose, testfilter='', rebuild=False, label=''):
    
    with open('runfiles/ip_piranha', 'w+') as f:
        for ip in ips:
            print(ip, file=f)

    with open('runfiles/base_config.json', 'r') as f:
        full_config = json.load(f)

    full_config.update(configuration)
    full_config['num_parties'] = nparties
    full_config['party_ips'] = ips
    full_config['party_users'] = ['ubuntu'] * nparties
    full_config['run_name'] = label
    full_config['network'] = network

    with open('runfiles/current_config.json', 'w+') as f:
        json.dump(full_config, f)

    cmd = 'ansible-playbook --private-key ~/.ssh/piranha-ae -i runfiles/hosts.yml -l {} runfiles/run_piranha.yml -e "conf=current_config.json" -e "ip_file=ip_piranha" -e "num_parties={}" -e "float_precision={}" -e "protocol={}" -e "label={}" -e "fig={}" -e "testfilter={}"'.format(
        target_hosts, 
        nparties,
        fp,
        protocol,
        label,
        figure,
        testfilter
    )

    if rebuild:
        cmd += ' -e "rebuild=1"'

    if not verbose:
        cmd += ' >/dev/null 2>&1'

    status = os.system(cmd)
    if os.waitstatus_to_exitcode(status) != 0:
        print('Piranha run failed with code {}'.format(os.waitstatus_to_exitcode(status)))
        exit(os.waitstatus_to_exitcode(status))


# ---- Figure 4 ----

fig4_FP = 26
fig4_protocols = [
    ('-DFOURPC', 4), # FantasticFour, the unit tests will cover all 3 protocols
]
fig4_mpspdz_protocols = [
    ('replicated-ring-party.x', 3),
    ('semi2k-party.x', 2),
    ('rep4-ring-party.x', 4)
]
fig4_config = {
    'run_unit_tests': True,
    'unit_test_only': True
}
fig4_testfilter = 'EvalTest*'

def run_fig4(ips, fast, verbose):
    # do nothing with `fast` because the microbenchmarks are already fast

    lan_ips, _, mpspdz_ips = ips

    # Run and retrieve Piranha benchmarks -> results/fig4
    for (protocol, num_parties) in tqdm(fig4_protocols, desc='Piranha microbenchmarks'):
        rebuild_piranha('lan', protocol, 26, verbose)
        run_piranha_experiment(
            'lan', lan_ips, protocol, num_parties, 26, fig4_config,
            MODEL_PATH+'secureml-norelu.json', 'fig4', verbose, testfilter=fig4_testfilter,
            rebuild=False, label='fig4-piranha-benchmark-{}pc'.format(num_parties)
        )

    # Run and retrieve mpspdz benchmarks -> results/fig4
    #  - create runfiles/ip_mpspdz
    #  - use ansible to run benchmarks
    with open('runfiles/ip_mpspdz', 'w+') as f:
        for ip in mpspdz_ips:
            print(ip, file=f)

    for (protocol, num_parties) in tqdm(fig4_mpspdz_protocols, desc='Figure 4 | Protocol'):
        for benchmark in tqdm(glob.glob('runfiles/mpspdz_bench*.mpc'), desc='Microbenchmarks'):
            benchmark = os.path.basename(os.path.splitext(benchmark)[0])

            if fast and benchmark == 'mpspdz_bench_conv_e': # if we're trying to go fast, ignore the really slow convolution
                continue

            args = ' -e "mpspdz_args=\\"-N 2\\""' if protocol == 'semi2k-party.x' else ' -e "mpspdz_args=\\"\\""'

            cmd = 'ansible-playbook --private-key ~/.ssh/piranha-ae -i runfiles/hosts.yml -l mpspdz runfiles/run_fig4_mpspdz.yml -e "protocol={}" -e "benchmark_name={}" -e "num_parties={}"'.format(protocol, benchmark, num_parties) + args
            #print(cmd)
            if not verbose:
                cmd += ' >/dev/null 2>&1'

            status = os.system(cmd)
            if os.waitstatus_to_exitcode(status) != 0:
                print('benchmark run failed with code {}'.format(os.waitstatus_to_exitcode(status)))
                exit(os.waitstatus_to_exitcode(status))


# ---- Figure 5 ----

fig5_FPs = [10, 12, 14, 16, 18, 20, 22, 24, 26]
fig5_networks = [
    MODEL_PATH+'secureml-norelu.json',
    MODEL_PATH+'lenet-norelu-avgpool.json',
    MODEL_PATH+'alexnet-cifar10-norelu.json',
    MODEL_PATH+'vgg16-cifar10-norelu.json'
]
fig5_protocol = ('', 3)    # 3PC/Falcon
fig5_config = {
    'custom_epochs': True,
    'custom_epoch_count': 10,
    'custom_batch_size': True,
    'custom_batch_size_count': 128,
    'no_test': True,
    'last_test': True,
    'eval_accuracy': True
}

def run_fig5(ips, fast, verbose):

    lan_ips, _, _ = ips

    # Run and retrieve Piranha training accuracies -> results/fig5
    for fp in tqdm(fig5_FPs, desc='Figure 5 | Fixed point precision'):
        rebuild_piranha('lan', fig5_protocol[0], fp, verbose)

        for network in tqdm(fig5_networks, desc='Network'):

            if fast and 'vgg16' in network:
                continue #skip the reaaaaally slow VGG if we're in a hurry

            run_piranha_experiment(
                'lan', lan_ips, fig5_protocol[0], fig5_protocol[1], fp, fig5_config,
                network, 'fig5', verbose, rebuild=False, label='fig5-{}-fp{}'.format(network.split('/')[-1], fp)
            )


# ---- Figure 6 ----

fig6_FP = 26
fig6_networks = [
    MODEL_PATH+'secureml-norelu.json',
    MODEL_PATH+'lenet-norelu-avgpool.json',
    MODEL_PATH+'alexnet-cifar10-norelu.json',
    MODEL_PATH+'vgg16-cifar10-norelu.json'
]
fig6_settings = ['lan', 'wan']
fig6_protocols = [
    ('-DTWOPC', 2), # SecureML
    ('', 3),    # 3PC/Falcon, default
    ('-DFOURPC', 4), # FantasticFour
]
fig6_config = {
    'custom_epochs': True,
    'custom_epoch_count': 1,
    'custom_iterations': True,
    'custom_iteration_count': 1,
    'custom_batch_size': True,
    'custom_batch_size_count': 1,
    'no_test': True,
    'eval_train_stats': True
}

def run_fig6(ips, fast, verbose):

    lan_ips, wan_ips, _ = ips

    # Run and retrieve Piranha communication/computation split for LAN
    for protocol, num_parties in tqdm(fig6_protocols, desc='Figure 6 | LAN Protocol'):
        break
        rebuild_piranha('lan', protocol, 26, verbose)
        for network in tqdm(fig6_networks, desc='Network'):

            if fast and 'vgg16' in network:
                continue #skip the slow VGG if we're in a hurry

            run_piranha_experiment(
                'lan', lan_ips, protocol, num_parties, 26, fig6_config,
                network, 'fig6', verbose, rebuild=False, label='fig6-lan-{}-{}party'.format(network.split('/')[-1], num_parties)
            )

    # Same thing for WAN
    for protocol, num_parties in tqdm(fig6_protocols, desc='Figure 6 | WAN Protocol'):
        rebuild_piranha('wan', protocol, 26, verbose)
        for network in tqdm(fig6_networks, desc='Network'):

            if fast and 'vgg16' in network:
                continue #skip the reaaaaally slow VGG if we're in a hurry

            run_piranha_experiment(
                'wan', wan_ips, protocol, num_parties, 26, fig6_config,
                network, 'fig6', verbose, rebuild=False, label='fig6-wan-{}-{}party'.format(network.split('/')[-1], num_parties)
            )


# ---- Figure 7 ----

fig7_branches = ['mem-footprint-naive', 'mem-footprint-iterator', 'mem-footprint-typing']
def run_fig7(ips, fast, verbose):

    lan_ips, _, _ = ips

    with open('runfiles/ip_piranha', 'w+') as f:
        for ip in lan_ips:
            print(ip, file=f)

    for branch in tqdm(fig7_branches, desc='Figure 7 | Memory footprint run'):

        cmd = 'ansible-playbook --private-key ~/.ssh/piranha-ae -i runfiles/hosts.yml -l lan runfiles/run_fig7.yml -e "branch={}" -e "label={}" -e "num_parties=3"'.format(branch, 'fig7-{}-footprint'.format(branch))
        if not verbose:
            cmd += ' >/dev/null 2>&1'

        status = os.system(cmd)
        if os.waitstatus_to_exitcode(status) != 0:
            print('memory footprint run failed with code {}'.format(os.waitstatus_to_exitcode(status)))
            exit(os.waitstatus_to_exitcode(status))


def run_figure(fig_number, ips, fast, verbose):
    if fig_number == 4:
        run_fig4(ips, fast, verbose)
    elif fig_number == 5:
        run_fig5(ips, fast, verbose)
    elif fig_number == 6:
        run_fig6(ips, fast, verbose)
    elif fig_number == 7:
        run_fig7(ips, fast, verbose)
    else:
        print('unrecognized figure number', fig_number)
        exit(1)


# ---- Table 2 ----

table2_FP = 26
table2_networks = [
    MODEL_PATH+'secureml-norelu.json',
    MODEL_PATH+'lenet-norelu-avgpool.json',
    MODEL_PATH+'alexnet-cifar10-norelu.json',
    MODEL_PATH+'vgg16-cifar10-norelu.json'
]
table2_protocols = [
    ('-DTWOPC', 2), # SecureML
    ('', 3),    # 3PC/Falcon, default
    ('-DFOURPC', 4), # FantasticFour
]
table2_config = {
    'custom_epochs': True,
    'custom_epoch_count': 10,
    'custom_batch_size': True,
    'custom_batch_size_count': 128,
    'no_test': True,
    'last_test': True,
    'eval_epoch_stats': True,
    'eval_accuracy': True
}

def run_table2(ips, fast, verbose):

    lan_ips, _, _ = ips

    # Run and retrieve Piranha communication/computation split for LAN
    for protocol, num_parties in tqdm(table2_protocols, desc='Table 2 | Protocol'):
        rebuild_piranha('lan', protocol, 26, verbose)
        for network in tqdm(table2_networks, desc='Network'):

            if fast and 'vgg16' in network:
                continue #skip the reaaaaally slow VGG if we're in a hurry

            run_piranha_experiment(
                'lan', lan_ips, protocol, num_parties, 26, table2_config,
                network, 'table2', verbose, rebuild=False, label='table2-{}-{}'.format(network.split('/')[-1], protocol)
            )


# ---- Table 3 ----

table3_FP = 26
table3_networks = [
    MODEL_PATH+'secureml-norelu.json',
    MODEL_PATH+'lenet-norelu-avgpool.json',
    MODEL_PATH+'alexnet-cifar10-norelu.json',
    MODEL_PATH+'vgg16-cifar10-norelu.json'
]
table3_protocols = [
    ('-DTWOPC', 2), # SecureML
    ('', 3),    # 3PC/Falcon, default
    ('-DFOURPC', 4), # FantasticFour
]
table3_batch_sizes = [1, 64, 128]
table3_config = {
    'custom_epochs': True,
    'custom_epoch_count': 1,
    'custom_iterations': True,
    'custom_iteration_count': 1,
    'no_test': True,
}

def run_table3(ips, fast, verbose):

    lan_ips, _, _ = ips

    # Run and retrieve Piranha communication/computation split for LAN
    for protocol, num_parties in tqdm(table3_protocols, desc='Table 3 | Protocol'):
        rebuild_piranha('lan', protocol, 26, verbose)
        for network in tqdm(table3_networks, desc='Network'):
            for k in tqdm(table3_batch_sizes, desc='Batch size'):

                table3_config['custom_batch_size'] = True
                table3_config['custom_batch_size_count'] = k

                label = 'table3-batch{}-{}-{}'.format(k, network.split('/')[-1], protocol)

                with open('runfiles/ip_piranha', 'w+') as f:
                    for ip in lan_ips:
                        print(ip, file=f)

                with open('runfiles/base_config.json', 'r') as f:
                    full_config = json.load(f)

                full_config.update(table3_config)
                full_config['num_parties'] = num_parties
                full_config['party_ips'] = lan_ips
                full_config['party_users'] = ['ubuntu'] * num_parties
                full_config['run_name'] = label
                full_config['network'] = network

                with open('runfiles/current_config.json', 'w+') as f:
                    json.dump(full_config, f)

                cmd = 'ansible-playbook --private-key ~/.ssh/piranha-ae -i runfiles/hosts.yml -l {} runfiles/run_table3.yml -e "conf=current_config.json" -e "ip_file=ip_piranha" -e "num_parties={}" -e "float_precision={}" -e "protocol={}" -e "label={}" -e "fig={}" -e "testfilter={}"'.format(
                    'lan',
                    num_parties,
                    26,
                    protocol,
                    label,
                    'table3',
                    '' 
                )

                if not verbose:
                    cmd += ' >/dev/null 2>&1'

                status = os.system(cmd)
                if os.waitstatus_to_exitcode(status) != 0:
                    print('Table 3 run failed with code {}'.format(os.waitstatus_to_exitcode(status)))
                    exit(os.waitstatus_to_exitcode(status))


# ---- Table 4 ----

table4_FP = 26
table4_networks = [
    MODEL_PATH+'lenet-norelu-avgpool.json',
    MODEL_PATH+'alexnet-cifar10-norelu.json',
    MODEL_PATH+'vgg16-cifar10-norelu.json'
]
table4_protocol = ('', 3)    # 3PC/Falcon, default
table4_inference_batch_size = 1
table4_training_batch_sizes = [128, 128, 32]
table4_config = {
    'custom_epochs': True,
    'custom_epoch_count': 1,
    'custom_iterations': True,
    'custom_iteration_count': 1,
    'no_test': True
}

def run_table4(ips, fast, verbose):

    lan_ips, _, _ = ips

    rebuild_piranha('lan', table4_protocol[0], 26, verbose)
    # Run and retrieve Piranha communication/computation split for LAN
    for i, network in tqdm(enumerate(table4_networks), desc='Table 4 | Network'):

        # Inference
        table4_config['custom_batch_size'] = True
        table4_config['custom_batch_size_count'] = table4_inference_batch_size

        table4_config['eval_inference_stats'] = True
        table4_config['eval_train_stats'] = False
        table4_config['inference_only'] = True 

        run_piranha_experiment(
            'lan', lan_ips, table4_protocol[0], table4_protocol[1], 26, table4_config,
            network, 'table4', verbose, rebuild=False, label='table4-inference-{}'.format(network.split('/')[-1])
        )
        run_piranha_experiment(
            'lan', lan_ips, table4_protocol[0], table4_protocol[1], 26, table4_config,
            network, 'table4', verbose, rebuild=False, label='table4-inference-{}'.format(network.split('/')[-1])
        )

        # Training
        table4_config['custom_batch_size'] = True
        table4_config['custom_batch_size_count'] = table4_training_batch_sizes[i] 

        table4_config['eval_inference_stats'] = False
        table4_config['eval_train_stats'] = True
        table4_config['inference_only'] = False

        run_piranha_experiment(
            'lan', lan_ips, table4_protocol[0], table4_protocol[1], 26, table4_config,
            network, 'table4', verbose, rebuild=False, label='table4-train-{}'.format(network.split('/')[-1])
        )


def run_table(tab_number, ips, fast, verbose):
    if tab_number == 2:
        run_table2(ips, fast, verbose)
    elif tab_number == 3:
        run_table3(ips, fast, verbose)
    elif tab_number == 4:
        run_table4(ips, fast, verbose)
    else:
        print('unrecognized table number', tab_number)
        exit(1)


def plot_fig4(data, outpath):

    fig_x = 2.6
    fig_y = fig_x/1.4

    marker_size = 4;
    small_marker_size = 3;
    legend_font_size = 7;
    axis_font_size = 6;

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('lines', linewidth=1.5, markersize=marker_size)
    #plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=axis_font_size)
    plt.rc('ytick', labelsize=axis_font_size)
    plt.rc('axes', labelsize=axis_font_size+2) 
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['xtick.major.size'] = 3
    mpl.rcParams['xtick.major.width'] = 0.4
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['ytick.major.width'] = 0.4

    figure_configs = [
        # ((p_series, p_marker, p_color, p_label), (m_series, m_marker, m_color, m_label))
        (('p-secureml', DEFAULT_MARKERS[0], CB_COLORS[0], "$\\bf{P-SecureML}$"), ('mpspdz-semi2k', DEFAULT_MARKERS[1], FADED_CB_COLORS[0], "MP-SPDZ semi2k")),
        (('p-falcon', DEFAULT_MARKERS[0], CB_COLORS[1], "$\\bf{P-Falcon}$"), ('mpspdz-repring', DEFAULT_MARKERS[1], FADED_CB_COLORS[1], "MP-SPDZ replicated-ring")),
        (('p-fantasticfour', DEFAULT_MARKERS[0], CB_COLORS[2], "$\\bf{P-FantasticFour}$"), ('mpspdz-rep4ring', DEFAULT_MARKERS[1], FADED_CB_COLORS[2], "MP-SPDZ rep4-ring")),
    ]

    for benchmark, xaxis_label in [('matrix-multiplication', 'Matrix dimensions (log scale)'), ('convolution', 'Convolution size (log scale)'), ('relu', 'ReLU size (log scale)')]:
        for p_cfg, m_cfg in figure_configs:

            p_series_label, p_marker, p_color, p_label = p_cfg
            p_series = data[benchmark][p_series_label]
            m_series_label, m_marker, m_color, m_label = m_cfg
            m_series = data[benchmark][m_series_label]

            fig, ax = plt.subplots(figsize=(fig_x, fig_y))   # width, height
            right_side = ax.spines["right"]
            right_side.set_visible(False)
            top_side = ax.spines["top"]
            top_side.set_visible(False)

            plt.plot(data[benchmark]['x'], p_series, '-', marker=p_marker, c=p_color, label=p_label, markersize=marker_size)
            plt.plot(data[benchmark]['x'], m_series, '--', linewidth=1.0, marker=m_marker, c=m_color, label=m_label, markersize=small_marker_size)

            plt.ylim(0.0005, 10000)
            plt.xlabel(xaxis_label)
            plt.ylabel('Run-time (sec, log scale)')
            plt.xscale('log')
            plt.yscale('log')
            if (benchmark == 'matrix-multiplication'):
                plt.xticks(data[benchmark]['x'])
                plt.gca().set_xticklabels(data[benchmark]['x'])

            plt.grid(alpha=0.3, axis='y') 
            legend = plt.legend(loc='upper left', fontsize=legend_font_size, handlelength=3)
            plt.tight_layout()
            plt.savefig(outpath+"fig4-{}-{}.png".format(benchmark, p_series_label), dpi=300, bbox_inches='tight')
            plt.close()


def plot_fig5(data, outpath):

    fig_x = 4.5
    fig_y = fig_x/1.4

    marker_size = 12;
    small_marker_size = 2;
    legend_font_size = 9;
    axis_font_size = 10;

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('lines', linewidth=1.2, markersize=marker_size)
    plt.rc('xtick', labelsize=axis_font_size)
    plt.rc('ytick', labelsize=axis_font_size)
    plt.rc('axes', labelsize=axis_font_size+1) 
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.4
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.width'] = 0.4
    mpl.rcParams['hatch.linewidth'] = 0.3

    marker = '.'
    colors = CB_COLORS

    fig, ax = plt.subplots(figsize=(fig_x, fig_y))   # width, height
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    plt.axhline(y=0.1, color='k', linestyle='--', linewidth=0.4)

    plt.plot(data['x'], data['secureml'], '-', marker=marker, c=colors[0], label="SecureML", markersize=marker_size)
    plt.plot(data['x'], data['lenet'], '-', marker=marker, c=colors[1], label="LeNet", markersize=marker_size)
    plt.plot(data['x'], data['alexnet'], '-', marker=marker, c=colors[2], label="AlexNet", markersize=marker_size)
    plt.plot(data['x'], data['vgg16'], '-', marker=marker, c=colors[3], label="VGG16", markersize=marker_size)

    plt.ylim(0.0, 1.0)
    plt.xlabel('Fixed Point Precision')
    plt.ylabel('Test Accuracy')
    plt.xticks(data['x'])

    plt.grid(alpha=0.3, axis='y') 
    legend = plt.legend(loc='upper left', fontsize=legend_font_size, handlelength=2)
    plt.tight_layout()
    plt.savefig(outpath+"fig5.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig6(data, outpath):

    fig_x = 4.5
    fig_y = fig_x/2

    marker_size = 12;
    small_marker_size = 2;
    legend_font_size = 9;
    axis_font_size = 10;

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('lines', linewidth=1.2, markersize=marker_size)
    plt.rc('xtick', labelsize=axis_font_size)
    plt.rc('ytick', labelsize=axis_font_size)
    plt.rc('axes', labelsize=axis_font_size+1) 
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.4
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.width'] = 0.4
    mpl.rcParams['hatch.linewidth'] = 0.3

    # [(secureml, lenet, alexnet, vgg)...] // for 2pc, 3pc, 4pc
    lan_zipped_comps = list(zip(data['lan']['secureml-computation'], data['lan']['lenet-computation'], data['lan']['alexnet-computation'], data['lan']['vgg16-computation']))
    lan_zipped_comms = list(zip(data['lan']['secureml-communication'], data['lan']['lenet-communication'], data['lan']['alexnet-communication'], data['lan']['vgg16-communication']))

    wan_zipped_comps = list(zip(data['wan']['secureml-computation'], data['wan']['lenet-computation'], data['wan']['alexnet-computation'], data['wan']['vgg16-computation']))
    wan_zipped_comms = list(zip(data['wan']['secureml-communication'], data['wan']['lenet-communication'], data['wan']['alexnet-communication'], data['wan']['vgg16-communication']))

    # chart

    labels = ["SecureML", "LeNet", "AlexNet", "VGG16"]
    colors = FADED_CB_COLORS
    faded_colors = FADED_CB_COLORS

    x = np.arange(4)
    width = 0.2

    fig, ax = plt.subplots(figsize=(fig_x, fig_y))   # width, height

    # LAN
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    rects1  = ax.bar(x - width, lan_zipped_comps[0], width, label="P-SecureML Comp.", color=colors[0], hatch='////', edgecolor='black', linewidth=0.3)
    rects1b = ax.bar(x - width, lan_zipped_comms[0], width, label="P-SecureML Comm.", bottom=lan_zipped_comps[0], color=colors[0], edgecolor='black', linewidth=0.3)
    rects2  = ax.bar(x, lan_zipped_comps[1], width, label="P-Falcon Comp.", color=colors[1], hatch='////', edgecolor='black', linewidth=0.3)
    rects2b = ax.bar(x, lan_zipped_comms[1], width, label="P-Falcon Comm.", bottom=lan_zipped_comps[1], color=colors[1], edgecolor='black', linewidth=0.3)
    rects3  = ax.bar(x + width, lan_zipped_comps[2], width, label="P-FantasticFour Comp.", color=colors[2], hatch='////', edgecolor='black', linewidth=0.3)
    rects3b = ax.bar(x + width, lan_zipped_comms[2], width, label="P-FantasticFour Comm.", bottom=lan_zipped_comps[2], color=colors[2], edgecolor='black', linewidth=0.3)

    ax.set_xlabel("LAN")
    ax.set_ylabel("Runtime (ms)")
    ax.set_xticks(x, labels)

    from matplotlib.patches import Patch

    pa1 = Patch(facecolor=colors[0], edgecolor='black', linewidth=0.3)
    pa2 = Patch(facecolor=colors[1], edgecolor='black', linewidth=0.3)
    pa3 = Patch(facecolor=colors[2], edgecolor='black', linewidth=0.3)
    pa4 = Patch(facecolor='white', edgecolor='black', linewidth=0.3)
    pa5 = Patch(facecolor='white', edgecolor='black', linewidth=0.3, hatch='////')

    plt.gca().legend(
                handles = [pa1, pa2, pa3, pa4, pa5],
                    labels = ['P-SecureML', 'P-Falcon', 'P-FantasticFour', 'Communication', 'Computation'],
                        ncol=1, handletextpad=0.5, handlelength=2, columnspacing=-0.5,
                            loc='upper left', fontsize=8
                            )
    plt.tight_layout()
    plt.savefig(outpath+"fig6-lan.png", dpi=300, bbox_inches='tight')
    plt.close()

    labels = ["SecureML", "LeNet", "AlexNet", "VGG16"]
    colors = FADED_CB_COLORS
    faded_colors = FADED_CB_COLORS

    x = np.arange(4)
    width = 0.2

    fig, ax = plt.subplots(figsize=(fig_x, fig_y))   # width, height

    # WAN
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    rects1  = ax.bar(x - width, wan_zipped_comps[0], width, label="P-SecureML Comp.", color=colors[0], hatch='////', edgecolor='black', linewidth=0.3)
    rects1b = ax.bar(x - width, wan_zipped_comms[0], width, label="P-SecureML Comm.", bottom=wan_zipped_comps[0], color=colors[0], edgecolor='black', linewidth=0.3)
    rects2  = ax.bar(x, wan_zipped_comps[1], width, label="P-Falcon Comp.", color=colors[1], hatch='////', edgecolor='black', linewidth=0.3)
    rects2b = ax.bar(x, wan_zipped_comms[1], width, label="P-Falcon Comm.", bottom=wan_zipped_comps[1], color=colors[1], edgecolor='black', linewidth=0.3)
    rects3  = ax.bar(x + width, wan_zipped_comps[2], width, label="P-FantasticFour Comp.", color=colors[2], hatch='////', edgecolor='black', linewidth=0.3)
    rects3b = ax.bar(x + width, wan_zipped_comms[2], width, label="P-FantasticFour Comm.", bottom=wan_zipped_comps[2], color=colors[2], edgecolor='black', linewidth=0.3)

    ax.set_xlabel("WAN")
    ax.set_xticks(x, labels)

    ax.set_ylabel("Runtime (ms)")
    ax.legend(fontsize=6, loc='upper left')

    pa1 = Patch(facecolor=colors[0], edgecolor='black', linewidth=0.3)
    pa2 = Patch(facecolor=colors[1], edgecolor='black', linewidth=0.3)
    pa3 = Patch(facecolor=colors[2], edgecolor='black', linewidth=0.3)
    pa4 = Patch(facecolor='white', edgecolor='black', linewidth=0.3)
    pa5 = Patch(facecolor='white', edgecolor='black', linewidth=0.3, hatch='////')

    plt.gca().legend(
                handles = [pa1, pa2, pa3, pa4, pa5],
                    labels = ['P-SecureML', 'P-Falcon', 'P-FantasticFour', 'Communication', 'Computation'],
                        ncol=1, handletextpad=0.5, handlelength=2, columnspacing=-0.5,
                            loc='upper left', fontsize=8
                            )

    plt.tight_layout()
    plt.savefig(outpath+"fig6-wan.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_fig7(df, outpath):

    fig_x = 4.5
    fig_y = fig_x/2

    marker_size = 12;
    small_marker_size = 2;
    legend_font_size = 9;
    axis_font_size = 10;

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('lines', linewidth=1.2, markersize=marker_size)
    plt.rc('xtick', labelsize=axis_font_size)
    plt.rc('ytick', labelsize=axis_font_size)
    plt.rc('axes', labelsize=axis_font_size+1) 
    mpl.rcParams['axes.linewidth'] = 0.4
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.4
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['ytick.major.width'] = 0.4
    mpl.rcParams['hatch.linewidth'] = 0.3

    benchmarks = [('naive', 'Naive Implementation', CB_COLORS[0]), ('iterator', 'Iterator-based Optimization', CB_COLORS[1]), ('typing', 'Type-based Optimization', CB_COLORS[2])]
    for b, b_label, b_color in benchmarks:

        timestamps = df['{} time (ms)'.format(b)]
        x = [i / len(timestamps) for i in range(len(timestamps))]

        y_bytes = df['{} (bytes)'.format(b)]
        y = [b / 1024.0 / 1024.0 for b in y_bytes]
    
        plt.ylim(0, 2500)

        plt.plot(x, y, '-', color=b_color, label=b_label)

        plt.xlabel('Computation Progress')
        plt.ylabel('GPU Memory Use (MB)')

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))

        plt.grid(alpha=0.3, axis='y') 
        legend = plt.legend(loc='upper left', fontsize=legend_font_size, handlelength=3)
        plt.tight_layout()
        plt.savefig(outpath+"fig7-{}.png".format(b), dpi=300, bbox_inches='tight')
        plt.close()


def generate_figures():
    
    # --------------- 
    # ---- Fig 4 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    fig4_raw_data = {
        'matrix-multiplication': {
            'x': [10, 30, 50, 100, 300],
            'p-secureml': [0] * 5,
            'mpspdz-semi2k': [0] * 5, 
            'p-falcon': [0] * 5, 
            'mpspdz-repring': [0] * 5, 
            'p-fantasticfour': [0] * 5, 
            'mpspdz-rep4ring': [0] * 5, 
        },
        'convolution': {
            'x': [14745600, 57600000, 270950400, 552960000, 1858560000],
            'p-secureml': [0] * 5,
            'mpspdz-semi2k': [0] * 5, 
            'p-falcon': [0] * 5, 
            'mpspdz-repring': [0] * 5, 
            'p-fantasticfour': [0] * 5, 
            'mpspdz-rep4ring': [0] * 5, 
        },
        'relu': {
            'x': [10, 100, 1000, 10000, 100000],
            'p-secureml': [0] * 5,
            'mpspdz-semi2k': [0] * 5, 
            'p-falcon': [0] * 5, 
            'mpspdz-repring': [0] * 5, 
            'p-fantasticfour': [0] * 5, 
            'mpspdz-rep4ring': [0] * 5, 
        }
    }

    # MP-SPDZ output
    setups = [
        ('matmul', 'matrix-multiplication', ['10', '30', '50', '100', '300']),
        ('conv', 'convolution', ['a', 'b', 'c', 'd', 'e']),
        ('relu', 'relu', ['10', '100', '1000', '10000', '100000']),
    ]
    for (src_name, dst_name, suffixes) in setups:
        for idx, suffix in enumerate(suffixes):
            for (protocol, num_parties) in [('mpspdz-semi2k', 2), ('mpspdz-repring', 3), ('mpspdz-rep4ring', 4)]:

                filepath = 'results/fig4/mpspdz1/home/ubuntu/MP-SPDZ/mpspdz_bench_{}_{}-{}-0.txt'.format(
                    src_name, suffix, num_parties
                )
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        for l in f.readlines():
                            m = re.match(r"^Time = (\d+\.\d+) seconds$", l.strip())
                            if m:
                                fig4_raw_data[dst_name][protocol][idx] = float(m.group(1))

    # Piranha output
    filepath = 'results/fig4/piranha1-lan/home/ubuntu/piranha/output/fig4-piranha-benchmark-4pc.out'
    if os.path.exists(filepath):
        fig4_piranha_lines = []
        with open(filepath, 'r') as f:
            fig4_piranha_lines = [l.strip() for l in f.readlines()]

        setups = [
            ('matmul', 'matrix-multiplication', ['(N=10)', '(N=30)', '(N=50)', '(N=100)', '(N=300)']),
            ('conv', 'convolution', ['(N=1, Iw/h=28, Din=1, Dout=16, f=5)', '(N=1, Iw/h=12, Din=20, Dout=50, f=3)', '(N=1, Iw/h=32, Din=3, Dout=50, f=7)', '(N=1, Iw/h=64, Din=3, Dout=32, f=5)', '(N=1, Iw/h=224, Din=3, Dout=64, f=5)']),
            ('relu', 'relu', ['(N=10)', '(N=100)', '(N=1000)', '(N=10000)', '(N=100000)']),
        ]
        for (protocol, num_parties) in [('p-secureml', 2), ('p-falcon', 3), ('p-fantasticfour', 4)]:
            for (src_label, dst_label, suffixes) in setups:
                for idx, suffix in enumerate(suffixes):

                    for line in fig4_piranha_lines:
                        pattern_str = r"^{}PC - {} {} - (\d+\.\d+) sec.$".format(num_parties, src_label, re.escape(suffix))
                        m = re.match(pattern_str, line)
                        if m:
                            fig4_raw_data[dst_label][protocol][idx] = float(m.group(1))
            
    with open('artifact_figures/artifact/fig4.json', 'w+') as f:
        json.dump(fig4_raw_data, f)
    
    # * create paper and artifact images in experiments/artifact_figures
    with open('artifact_figures/artifact/fig4.json', 'r') as f:
        fig4_artifact = json.load(f)

    with open('artifact_figures/paper/fig4.json', 'r') as f:
        fig4_paper = json.load(f)

    plot_fig4(fig4_paper, 'artifact_figures/paper/')
    plot_fig4(fig4_artifact, 'artifact_figures/artifact/')


    # --------------- 
    # ---- Fig 5 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    fig5_raw_data = {
        'x': [10, 12, 14, 16, 18, 20, 22, 24, 26],
        'secureml': [0.0] * 9,
        'lenet': [0.0] * 9,
        'alexnet': [0.0] * 9,
        'vgg16': [0.0] * 9
    }

    models = [('secureml', 'secureml-norelu.json'), ('lenet', 'lenet-norelu-avgpool.json'), ('alexnet', 'alexnet-cifar10-norelu.json'), ('vgg16', 'vgg16-cifar10-norelu.json')]
    fps = [10, 12, 14, 16, 18, 20, 22, 24, 26]
    folder = 'results/fig5/piranha1-lan/home/ubuntu/piranha/output/'

    for dst_model, src_model in models:
        for idx, fp in enumerate(fps):
            filepath = folder + 'fig5-{}-fp{}.out'.format(src_model, fp)

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    fig5_piranha_lines = [l.strip() for l in f.readlines()]

                    for line in fig5_piranha_lines:
                        pattern_str = r"test accuracy,(\d+\.\d+)$"
                        m = re.match(pattern_str, line)
                        if m:
                            fig5_raw_data[dst_model][idx] = float(m.group(1))
            
    with open('artifact_figures/artifact/fig5.json', 'w+') as f:
        json.dump(fig5_raw_data, f)
    
    # * create paper and artifact images in experiments/artifact_figures
    with open('artifact_figures/artifact/fig5.json', 'r') as f:
        fig5_artifact = json.load(f)

    with open('artifact_figures/paper/fig5.json', 'r') as f:
        fig5_paper = json.load(f)

    plot_fig5(fig5_paper, 'artifact_figures/paper/')
    plot_fig5(fig5_artifact, 'artifact_figures/artifact/')


    # --------------- 
    # ---- Fig 6 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    fig6_raw_data = {
        'lan': {
            'secureml-computation': [0.0] * 3,
            'secureml-communication': [0.0] * 3,
            'lenet-computation': [0.0] * 3,
            'lenet-communication': [0.0] * 3,
            'alexnet-computation': [0.0] * 3,
            'alexnet-communication': [0.0] * 3,
            'vgg16-computation': [0.0] * 3,
            'vgg16-communication': [0.0] * 3,
        },
        'wan': {
            'secureml-computation': [0.0] * 3,
            'secureml-communication': [0.0] * 3,
            'lenet-computation': [0.0] * 3,
            'lenet-communication': [0.0] * 3,
            'alexnet-computation': [0.0] * 3,
            'alexnet-communication': [0.0] * 3,
            'vgg16-computation': [0.0] * 3,
            'vgg16-communication': [0.0] * 3,
        }
    }

    models = [('secureml', 'secureml-norelu.json'), ('lenet', 'lenet-norelu-avgpool.json'), ('alexnet', 'alexnet-cifar10-norelu.json'), ('vgg16', 'vgg16-cifar10-norelu.json')]
    settings = ['lan', 'wan']
    protocols = ['2party', '3party', '4party']

    for s in settings:
        folder = 'results/fig6/piranha1-{}/home/ubuntu/piranha/output/'.format(s)

        for dst_model, src_model in models:
            for idx, protocol in enumerate(protocols):
                filepath = folder + 'fig6-{}-{}-{}.out'.format(s, src_model, protocol)

                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        fig6_piranha_lines = [l.strip() for l in f.readlines()]

                        for line in fig6_piranha_lines:
                            pattern_str = r"^training computation \(ms\),(\d+\.\d+)$"
                            m = re.match(pattern_str, line)
                            if m:
                                fig6_raw_data[s][dst_model+'-computation'][idx] = float(m.group(1))

                            pattern_str = r"^training comm \(ms\),(\d+\.\d+)$"
                            m = re.match(pattern_str, line)
                            if m:
                                fig6_raw_data[s][dst_model+'-communication'][idx] = float(m.group(1))
                
    with open('artifact_figures/artifact/fig6.json', 'w+') as f:
        json.dump(fig6_raw_data, f)
    
    # * create paper and artifact images in experiments/artifact_figures
    with open('artifact_figures/artifact/fig6.json', 'r') as f:
        fig6_artifact = json.load(f)

    with open('artifact_figures/paper/fig6.json', 'r') as f:
        fig6_paper = json.load(f)

    plot_fig6(fig6_paper, 'artifact_figures/paper/')
    plot_fig6(fig6_artifact, 'artifact_figures/artifact/')


    # --------------- 
    # ---- Fig 7 ---- 
    # --------------- 

    # * this one is different, we're just trying to replicate a csv of the figure input

    df = pd.DataFrame()

    benchmarks = ['naive', 'iterator', 'typing']
    folder = 'results/fig7/piranha1-lan/home/ubuntu/piranha/'
    for b in benchmarks:

        filepath = folder + 'fig7-mem-footprint-{}-footprint.out'.format(b)

        timestamps = []
        mems = []

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                fig7_piranha_lines = [l.strip() for l in f.readlines()]

                for line in fig7_piranha_lines:
                    pattern_str = r"^MEM,(\d+),(\d+)$"
                    m = re.match(pattern_str, line)
                    if m:
                        timestamps.append(int(m.group(1)))
                        mems.append(int(m.group(2)))

        new_columns = pd.DataFrame({
            '{} time (ms)'.format(b): timestamps,
            '{} (bytes)'.format(b): mems
        }, dtype=int)

        df = pd.concat([df, new_columns], axis=1)

    df.to_csv('artifact_figures/artifact/fig7.csv', index=False)
                
    # * create paper and artifact images in experiments/artifact_figures
    artifact_df = pd.read_csv('artifact_figures/artifact/fig7.csv')
    paper_df = pd.read_csv('artifact_figures/paper/fig7.csv')

    plot_fig7(paper_df, 'artifact_figures/paper/')
    plot_fig7(artifact_df, 'artifact_figures/artifact/')


def generate_tables():

    # --------------- 
    # ---- Tab 2 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    tab2_raw_data = {
        'secureml': {
            # time, comm, train accuracy, test accuracy
            "__desc": ["time", "comm", "train accuracy", "test accuracy"],
            'p-secureml': [0.0] * 4,
            'p-falcon': [0.0] * 4,
            'p-fantasticfour': [0.0] * 4
        },
        'lenet': {
            'p-secureml': [0.0] * 4,
            'p-falcon': [0.0] * 4,
            'p-fantasticfour': [0.0] * 4
        },
        'alexnet': {
            'p-secureml': [0.0] * 4,
            'p-falcon': [0.0] * 4,
            'p-fantasticfour': [0.0] * 4
        },
        'vgg16': {
            'p-secureml': [0.0] * 4,
            'p-falcon': [0.0] * 4,
            'p-fantasticfour': [0.0] * 4
        }
    }

    models = [('secureml', 'secureml-norelu.json'), ('lenet', 'lenet-norelu-avgpool.json'), ('alexnet', 'alexnet-cifar10-norelu.json'), ('vgg16', 'vgg16-cifar10-norelu.json')]
    protocols = [('p-secureml', 2, '-DTWOPC'), ('p-falcon', 3, ''), ('p-fantasticfour', 4, '-DFOURPC')]

    for dst_model, src_model in models:
        for dst_protocol, num_parties, src_protocol in protocols:
            
            comms = [0.0] * num_parties

            for p in range(num_parties):

                filepath = 'results/table2/piranha{}-lan/home/ubuntu/piranha/output/table2-{}-{}.out'.format(p, src_model, src_protocol)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        table2_piranha_lines = [l.strip() for l in f.readlines()]

                        for line in table2_piranha_lines:
                            pattern_str = r"^total tx comm \(MB\),(\d+\.\d+)$"
                            m = re.match(pattern_str, line)
                            if m:
                                comms.append(float(m.group(1)))

            filepath = 'results/table2/piranha1-lan/home/ubuntu/piranha/output/table2-{}-{}.out'.format(src_model, src_protocol)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    table2_piranha_lines = [l.strip() for l in f.readlines()]

                    time, train_acc, test_acc = 0.0, 0.0, 0.0

                    for line in table2_piranha_lines:

                        pattern_str = r"^total time \(s\),(\d+\.\d+)$"
                        m = re.match(pattern_str, line)
                        if m:
                            time = float(m.group(1))

                        pattern_str = r"^train accuracy,(\d+\.\d+)$"
                        m = re.match(pattern_str, line)
                        if m:
                            train_acc = float(m.group(1))

                        pattern_str = r"^test accuracy,(\d+\.\d+)$"
                        m = re.match(pattern_str, line)
                        if m:
                            test_acc = float(m.group(1))
                
                    tab2_raw_data[dst_model][dst_protocol] = [time, sum(comms) / num_parties, train_acc * 100, test_acc * 100]
    
    with open('artifact_figures/artifact/table2.json', 'w+') as f:
        json.dump(tab2_raw_data, f, indent=4)
    

    # --------------- 
    # ---- Tab 3 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    tab3_raw_data = {
        'secureml': {
            '__desc': ['p-secureml', 'p-falcon', 'p-fantasticfour'],
            1: [0] * 3,
            64: [0] * 3,
            128: [0] * 3,
        },
        'lenet': {
            1: [0] * 3,
            64: [0] * 3,
            128: [0] * 3,
        },
        'alexnet': {
            1: [0] * 3,
            64: [0] * 3,
            128: [0] * 3,
        },
        'vgg16': {
            1: [0] * 3,
            64: [0] * 3,
            128: [0] * 3,
        }
    }

    models = [('secureml', 'secureml-norelu.json'), ('lenet', 'lenet-norelu-avgpool.json'), ('alexnet', 'alexnet-cifar10-norelu.json'), ('vgg16', 'vgg16-cifar10-norelu.json')]
    protocols = [('p-secureml', 2, '-DTWOPC'), ('p-falcon', 3, ''), ('p-fantasticfour', 4, '-DFOURPC')]

    for dst_model, src_model in models:
        for dst_protocol, num_parties, src_protocol in protocols:
            for k in [1, 64, 128]:
            
                filepath = 'results/table3/piranha1-lan/home/ubuntu/piranha/output/table3-batch{}-{}-{}.out'.format(k, src_model, src_protocol)
                if os.path.exists(filepath):
                    max_mem_mb = float(subprocess.check_output('sort -g -k 1 {} | tail -n 1'.format(filepath), shell=True).split()[0])
                    tab3_raw_data[dst_model][k][num_parties - 2] = max_mem_mb
    
    with open('artifact_figures/artifact/table3.json', 'w+') as f:
        json.dump(tab3_raw_data, f, indent=4)


    # --------------- 
    # ---- Tab 4 ---- 
    # --------------- 

    # * create data file from raw output, replacing missing values with 0
    tab4_raw_data = {
        'time': {
            "__desc": ["falcon-inference", "cryptgpu-inference", "piranha-inference", "falcon-training", "cryptgpu-training", "piranha-training"],
            "lenet": [0.038, 0.380, 0.0, 14.9, 2.21, 0.0],
            "alexnet": [0.110, 0.910, 0.0, 62.37, 2.910, 0.0],
            "vgg16": [1.440, 2.140, 0.0, 360.83, 12.140, 0.0]
        },
        'comm': {
            "lenet": [2.29, 3, 0.0, 0.346, 1.14, 0.0],
            "alexnet": [4.02, 2.43, 0.0, 0.621, 1.37, 0.0],
            "vgg16": [40.05, 56.2, 0.0, 1.78, 7.55, 0.0]
        },
    }

    models = [('lenet', 'lenet-norelu-avgpool.json'), ('alexnet', 'alexnet-cifar10-norelu.json'), ('vgg16', 'vgg16-cifar10-norelu.json')]
    modes = ['inference', 'train']

    for dst_model, src_model in models:
        for mode in modes:
            
            comms = [0.0] * 3
            for p in range(3):

                filepath = 'results/table4/piranha{}-lan/home/ubuntu/piranha/output/table4-{}-{}.out'.format(p, mode, src_model)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        table4_piranha_lines = [l.strip() for l in f.readlines()]

                        for line in table4_piranha_lines:
                            if mode == 'inference':
                                pattern_str = r"^inference TX comm \(bytes\),(\d+)$"
                            else:
                                pattern_str = r"^training TX comm \(MB\),(\d+\.\d+)$"

                            m = re.match(pattern_str, line)
                            if m:
                                if mode == 'inference':
                                    comms.append(float(int(m.group(1))/1024.0/1024.0))
                                else: # Training -> GB
                                    comms.append(float(m.group(1))/1024.0)

            tab4_raw_data['comm'][dst_model][2 if mode == 'inference' else 5] = sum(comms)

            filepath = 'results/table4/piranha1-lan/home/ubuntu/piranha/output/table4-{}-{}.out'.format(mode, src_model)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    table4_piranha_lines = [l.strip() for l in f.readlines()]

                    time = 0.0

                    for line in table4_piranha_lines:

                        if mode == 'inference':
                            pattern_str = r"^inference iteration \(ms\),(\d+\.\d+)$"
                        else:
                            pattern_str = r"^training iteration \(ms\),(\d+\.\d+)$"

                        m = re.match(pattern_str, line)
                        if m:
                            time = float(m.group(1)) / 1000.0

                    tab4_raw_data['time'][dst_model][2 if mode == 'inference' else 5] = time

    with open('artifact_figures/artifact/table4.json', 'w+') as f:
        json.dump(tab4_raw_data, f, indent=4)


def main():

    parser = argparse.ArgumentParser(description='Run artifact evaluation!')
    parser.add_argument('--start', default=False, action='store_true', help='Provision cluster for experiments. _Please suspend the cluster while not running experiments :)_')
    parser.add_argument('--stop', default=False, action='store_true', help='Suspend evaluation machines.')
    parser.add_argument('--figure', default=None, type=int, help='Figure # to run.')
    parser.add_argument('--table', default=None, type=int, help='Table # to run.')
    parser.add_argument('--generate', default=False, action='store_true', help='Generate figure/table images.')
    parser.add_argument('--fast', default=False, action='store_true', help='Run all the (relatively) fast runs, see README for more information')
    parser.add_argument('--verbose', default=False, action='store_true', help='Display verbose run commands, helpful for debugging')

    args = parser.parse_args();

    # Provision cluster
    if args.start:
        start_machines()

    # Suspend cluster
    elif args.stop:
        suspend_machines()

    # Handle figure experiments
    elif args.figure:
        ips = update_hosts()
        run_figure(args.figure, ips, args.fast, args.verbose)
        
    # Handle tables
    elif args.table:
        ips = update_hosts()
        run_table(args.table, ips, args.fast, args.verbose)

    elif not args.generate:
        parser.print_help()
        exit(1)

    if args.generate:
        generate_figures()
        generate_tables()


if __name__ == '__main__':
    main();

