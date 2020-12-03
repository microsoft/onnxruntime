#! /usr/bin/env bash
sudo mkdir /bert_data
if [ ! -d "/etc/smbcredentials" ]; then
sudo mkdir /etc/smbcredentials
fi
if [ ! -f "/etc/smbcredentials/onnxruntimetestdata.cred" ]; then
    sudo bash -c 'echo "username=onnxruntimetestdata" >> /etc/smbcredentials/onnxruntimetestdata.cred'
    sudo bash -c 'echo "password=$1" >> /etc/smbcredentials/onnxruntimetestdata.cred'
fi
sudo chmod 600 /etc/smbcredentials/onnxruntimetestdata.cred

sudo bash -c 'echo "//onnxruntimetestdata.file.core.windows.net/bert-data /bert_data cifs nofail,vers=3.0,credentials=/etc/smbcredentials/onnxruntimetestdata.cred,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab'
sudo mount -t cifs //onnxruntimetestdata.file.core.windows.net/bert-data /bert_data -o vers=3.0,credentials=/etc/smbcredentials/onnxruntimetestdata.cred,dir_mode=0777,file_mode=0777,serverino
