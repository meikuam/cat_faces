import os
import time
import subprocess

""" remote PC script """
project_name = 'cat_faces'
host = 'localhost'
user = 'user'

jupyter_port = '8881'
visible_device = '0'
data_folder = 'data'
project_root = os.path.join('/home/', user, project_name)
container = project_name + '-dev-instance-0'

check_if_running = os.popen('docker ps -f name=' + container + ' | tail -1 | grep ' + container).read().strip()
if not check_if_running:
    check_if_exists = os.popen('docker ps -f name=' + container + ' -a | tail -1 | grep ' + container).read().strip()
    if check_if_exists:
        print('docker container: * exists * ')
        subprocess.call('docker start ' + container, shell=True)
        print('docker container: now running ')
    else:
        print('docker container: * not exists * ')
        print('build container')
        subprocess.call('docker build -t ' + project_name +
                        ':dev docker/dev --build-arg UID=$(id -u $(whoami))' +
                        ' --build-arg USER=' + user +
                        ' --build-arg JUPUTER_PORT=' + jupyter_port +
                        ' --build-arg PROJECT_NAME=' + project_name
                        , shell=True)
        subprocess.call('docker run -it -d ' +
                        # ' -p ' + ssh_port + ':22 -p ' + jupyter_port + ':' + jupyter_port +
                        ' -P --name=' + container +
                        ' --runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=' + visible_device
                        +' -v $(readlink -e .):' + project_root +
                        ' -v ' + data_folder + ':' + project_root + '/data' +
                        ' ' + project_name + ':dev bash', shell=True)
        print('docker container: now running ')

        print('start ssh service')
        subprocess.call('docker exec --user root ' + container + ' service ssh start', shell=True)
else:
    print('docker container: already running')



print('start jupyter')
subprocess.call('docker exec --user ' + user + ' ' + container +
                ' jupyter notebook --ip=0.0.0.0 --port ' + jupyter_port +
                ' --no-browser > /dev/null 2>&1 &',
                shell=True)

time.sleep(2)
jupyter_token = os.popen("docker exec " + container + " jupyter notebook list").read()
print(jupyter_token)
jupyter_token = jupyter_token.split('::')[0].strip().split('?token=')[-1]

ssh_port_host = os.popen("docker port " + container + " 22").read().split(':')[-1].strip()
jupyter_port_host = os.popen("docker port " + container + " " + jupyter_port).read().split(':')[-1].strip()


jupyter_link = 'http://' + host + ':' + jupyter_port_host + '/tree?token=' + jupyter_token
print('jupyter link: ', jupyter_link)

os.system('ssh ' + user + '@' + host + ' -p ' + ssh_port_host)
