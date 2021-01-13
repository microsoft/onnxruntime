#! /usr/bin/env bash
if [ -d "/bert_data" ]; then
sudo umount /bert_data
fi
if [ -d "/bert_data" ]; then
sudo rm -rf /bert_data
fi

sudo mkdir /bert_data

if [ ! -d "/etc/smbcredentials" ]; then
sudo mkdir /etc/smbcredentials
fi

if [ -f "/etc/smbcredentials/orttrainingtestdata.cred" ]; then
sudo rm /etc/smbcredentials/orttrainingtestdata.cred
fi

# to create orttrainingtestdata.cred, I have to do: 'sudo bash -c ...'
sudo bash -c 'echo "username=orttrainingtestdata" >> /etc/smbcredentials/orttrainingtestdata.cred'

# $1 get removed (do defend injection attack?) if I do 'sudo bash -c...'
# to enable 'sudo echo...' I need to 'sudo chmod 777...' first.
sudo chmod 777 /etc/smbcredentials/orttrainingtestdata.cred
sudo echo "password=$1" >> /etc/smbcredentials/orttrainingtestdata.cred

sudo chmod 600 /etc/smbcredentials/orttrainingtestdata.cred

sudo bash -c 'echo "//orttrainingtestdata.file.core.windows.net/bert-data /bert_data cifs nofail,vers=3.0,credentials=/etc/smbcredentials/orttrainingtestdata.cred,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab'
sudo mount -t cifs //orttrainingtestdata.file.core.windows.net/bert-data /bert_data -o vers=3.0,credentials=/etc/smbcredentials/orttrainingtestdata.cred,dir_mode=0777,file_mode=0777,serverino
