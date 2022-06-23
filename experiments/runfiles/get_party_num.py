
import argparse
import json

parser = argparse.ArgumentParser(description='Extract party number for IP')
parser.add_argument('external_ip', type=str)
parser.add_argument('file', type=str)
args = parser.parse_args()

found = False
with open(args.file, 'r') as f:

    ips = [x.strip() for x in f.readlines()]
    for i, ip in enumerate(ips):
        if ip == args.external_ip:
            print(i)
            found = True
            break

    '''
    conf = json.load(f)

    for i, ip in enumerate(conf['party_ips']):
        if ip == args.external_ip:
            print(i)
            found = True
            break
    '''

exit(0 if found else 1)

