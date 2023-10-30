from re import I
import subprocess
import time
import sys

pid = sys.argv[1]
cmd = sys.argv[2]
step = int(sys.argv[3]) * 60  # unit is min
while True:
    follow_command = "ps -ef | awk '{if($2==" + pid + ") print $0}' | wc -l"
    process = subprocess.run(follow_command, shell=True, stdout=subprocess.PIPE)
    # print(process)
    running_flag = int(process.stdout.decode('utf8').strip())
    if not running_flag :
        subprocess.run(cmd, shell=True)
        break
    else:
        print("the before process is running now")
        time.sleep(step)