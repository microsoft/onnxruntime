#! /usr/bin/env bash
mkdir /bert_data
if [ ! -d "/etc/smbcredentials" ]; then
mkdir /etc/smbcredentials
fi
if [ ! -f "/etc/smbcredentials/onnxruntimetestdata.cred" ]; then
    bash -c 'echo "username=onnxruntimetestdata" >> /etc/smbcredentials/onnxruntimetestdata.cred'
    bash -c 'echo "password=PHlXrgqZjn8iw8pMlBgD3DSN95+A5bX2ZiNOx1+yNJ8GasYJT+zZpDzPHD4nlrIFA00un7InSjiCstvItXpBuA==" >> /etc/smbcredentials/onnxruntimetestdata.cred'
fi
chmod 600 /etc/smbcredentials/onnxruntimetestdata.cred

bash -c 'echo "//onnxruntimetestdata.file.core.windows.net/bert-data /bert_data cifs nofail,vers=3.0,credentials=/etc/smbcredentials/onnxruntimetestdata.cred,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab'
mount -t cifs //onnxruntimetestdata.file.core.windows.net/bert-data /bert_data -o vers=3.0,credentials=/etc/smbcredentials/onnxruntimetestdata.cred,dir_mode=0777,file_mode=0777,serverino
