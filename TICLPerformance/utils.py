import os
import subprocess

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def deploy_php(path):
    filephp = os.path.join(path + "/index.php")
    if(not os.path.exists(filephp)):
      command = "pb_copy_index.py " + path
      subprocess.run(command, shell = True)

