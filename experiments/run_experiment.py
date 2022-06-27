
import argparse
import boto3
import glob
import json
import os
import time
from tqdm import tqdm

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

    with open('runfiles/hosts.yml', 'w') as f:
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
    
    with open('runfiles/ip_piranha', 'w') as f:
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

    with open('runfiles/current_config.json', 'w') as f:
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
    with open('runfiles/ip_mpspdz', 'w') as f:
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

    with open('runfiles/ip_piranha', 'w') as f:
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

                with open('runfiles/ip_piranha', 'w') as f:
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

                with open('runfiles/current_config.json', 'w') as f:
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


def main():

    parser = argparse.ArgumentParser(description='Run artifact evaluation!')
    parser.add_argument('--start', default=False, action='store_true', help='Provision cluster for experiments. _Please suspend the cluster while not running experiments :)_')
    parser.add_argument('--stop', default=False, action='store_true', help='Suspend evaluation machines.')
    parser.add_argument('--figure', default=None, type=int, help='Figure # to run.')
    parser.add_argument('--table', default=None, type=int, help='Table # to run.')
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

    else:
        parser.print_help()
        exit(1)


if __name__ == '__main__':
    main();

