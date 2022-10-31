
import argparse
import os
import sys
import subprocess

from pathlib import Path

def run1(c, cwd=None):
    if isinstance(c, list):
        print(' '.join(c))
        subprocess.check_call(c)
    else:
        print(c)
        subprocess.check_call(c, shell=True, cwd=cwd)

def go(gpu, argv):

    dockerfile = 'Dockerfile.cpu' if not gpu else 'Dockerfile.trt'
    target = 'octoml/olive:cpu' if not gpu else 'octoml/olive:trt'

    run1(f"docker build -t {target} -f {dockerfile} .")
    
    home = str(Path.home())
    config_dir = os.path.join(os.getcwd(), "configs")
    args = ['docker', 'run', '-it', '--rm', '--volume', f'{home}:/tmp', '--volume', f'{config_dir}:/configs']

    if gpu:
        args.extend(['--gpus', 'all'])
    
    aws_keys = ['AWS_ACCESS_KEY_ID', 'AWS_SESSION_TOKEN','AWS_SECRET_ACCESS_KEY']
    for key in aws_keys:
        args.extend(['--env', key])

    args.append(target)
    args.extend(argv)
    run1(args)

if __name__ == '__main__':
    gpu = '-g' in sys.argv or '--gpu' in sys.argv
    go(gpu, sys.argv[1:])
