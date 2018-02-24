#!/usr/bin/env bash

# aws ec2 describe-images --filters Name=name,Values="Deep Learning Base AMI (Ubuntu) Version 3.0"

echo "creating machine"
docker-machine create \
    --driver amazonec2 \
    --amazonec2-access-key $ACCESS_KEY \
    --amazonec2-secret-key $SECRET_KEY \
    --amazonec2-region us-east-1 \
    --amazonec2-instance-type p2.xlarge \
    --amazonec2-ami ami-4f1d2a35 \
    --amazonec2-root-size 50 \
    --amazonec2-ssh-user ubuntu \
    --amazonec2-open-port 9999 \
    --engine-storage-driver overlay2 \
    --engine-install-url https://raw.githubusercontent.com/rancher/install-docker/master/17.12.0.sh \
    $MACHINE_NAME
echo "$(docker-machine ip $MACHINE_NAME)"


echo "hack to reprovision"
sleep 30s
docker-machine ssh $MACHINE_NAME "sudo rm /var/lib/apt/lists/lock"
docker-machine ssh $MACHINE_NAME "sudo rm /var/cache/apt/archives/lock"
docker-machine ssh $MACHINE_NAME "sudo rm /var/lib/dpkg/lock"
docker-machine provision $MACHINE_NAME

echo "securing machine"
docker-machine ssh $MACHINE_NAME "sudo apt-get update"
docker-machine ssh $MACHINE_NAME "sudo apt-get -y install fail2ban"
docker-machine ssh $MACHINE_NAME "sudo ufw default deny"
docker-machine ssh $MACHINE_NAME "sudo ufw allow ssh"
docker-machine ssh $MACHINE_NAME "sudo ufw allow http"
docker-machine ssh $MACHINE_NAME "sudo ufw allow 2376" # Docker
docker-machine ssh $MACHINE_NAME "sudo ufw --force enable"

echo "installing nvidia docker runtime"
# Add the package repositories
docker-machine ssh $MACHINE_NAME "sudo curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
docker-machine ssh $MACHINE_NAME "sudo curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
docker-machine ssh $MACHINE_NAME "sudo apt-get update"

# Install nvidia-docker2 and reload the Docker daemon configuration
docker-machine ssh $MACHINE_NAME "sudo apt-get install -y nvidia-docker2"
docker-machine ssh $MACHINE_NAME 'sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF'
docker-machine ssh $MACHINE_NAME "sudo pkill -SIGHUP dockerd"

eval "$(docker-machine env $MACHINE_NAME)"
echo $DOCKER_HOST
echo "building"
docker-compose -H $DOCKER_HOST build
echo "starting up"
docker-compose -H $DOCKER_HOST up