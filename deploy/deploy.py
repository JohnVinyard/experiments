import argparse
import shutil
import subprocess
import os
import json
from collections import namedtuple
import zounds
import glob

parser = argparse.ArgumentParser(
    parents=[
        zounds.ObjectStorageSettings(),
        zounds.AppSettings()
    ])

experiment = parser.add_argument_group(
    'experiment',
    'Experiment settings')
experiment.add_argument(
    '--name',
    help='the name of the experiment',
    required=True)
experiment.add_argument(
    '--path',
    help='absolute or relative path to experiment python script',
    required=True)

aws = parser.add_argument_group(
    'aws', 'AWS EC2 settings')
aws.add_argument(
    '--amazonec2-access-key',
    help='aws access key',
    required=True)
aws.add_argument(
    '--amazonec2-secret-key',
    help='aws secret key',
    required=True)


def get_args():
    try:
        with open('settings.json', 'r') as f:
            d = json.load(f)
            Settings = namedtuple('Settings', d.keys())
            print 'Settings loaded from settings.json'
            settings = Settings(**d)
            print settings
            return settings
    except (ValueError, IOError):
        return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    files = glob.glob1(args.path, '*.py')
    for f in files:
        shutil.copyfile(os.path.join(args.path, f), f)

    env = {
        'ACCESS_KEY': args.amazonec2_access_key,
        'SECRET_KEY': args.amazonec2_secret_key,
        'MACHINE_NAME': args.name,
        'OBJECT_STORAGE_REGION': args.object_storage_region,
        'OBJECT_STORAGE_USER': args.object_storage_username,
        'OBJECT_STORAGE_API_KEY': args.object_storage_api_key,
        'APP_SECRET': args.app_secret
    }
    abspath = os.path.abspath(__file__)
    path, script = os.path.split(abspath)
    deploy_script = os.path.join(path, 'deploy.sh')
    env.update(os.environ)
    subprocess.check_call([deploy_script], env=env)
