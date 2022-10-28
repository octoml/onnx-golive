import argparse
import os
import sys
import subprocess

def run1(c, cwd=None):
    if isinstance(c, list):
        print(' '.join(c))
        subprocess.check_call(c)
    else:
        print(c)
        subprocess.check_call(c, shell=True, cwd=cwd)

def go(gpu, argv):
    run1("sudo snap install docker")

    if not os.path.exists('onnx-golive'):
        run1("git clone https://github.com/octoml/onnx-golive.git")

    dockerfile = 'Dockerfile.cpu' if not gpu else 'Dockerfile.trt'
    target = 'octoml/olive:cpu' if not gpu else 'octoml/olive:trt'

    run1(f"sudo docker build -t {target} -f {dockerfile} .", cwd='onnx-golive')

    aws_keys = ['AWS_ACCESS_KEY_ID', 'AWS_SESSION_TOKEN','AWS_SECRET_ACCESS_KEY']

    args = ['sudo', f"--preserve-env={','.join(aws_keys)}",
            'docker', 'run', '-it', '--rm', '--volume', '/tmp:/tmp']
    for key in aws_keys:
        args.extend(['--env', key])
    args.append(target)
    args.extend(argv)
    run1(args)

if __name__ == '__main__':
    gpu = '-g' in sys.argv or '--gpu' in sys.argv
    go(gpu, sys.argv[1:])
