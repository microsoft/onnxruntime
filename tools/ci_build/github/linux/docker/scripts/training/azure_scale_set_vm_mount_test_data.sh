#! /usr/bin/env bash

function credentialize () {
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
}

function mount_data () {
    echo "Mounting source $1 at destination $2"

    if [ -d $2 ]; then
        sudo umount $2
        if [ $? != 0 ]; then
            echo "umount failed"
        fi
    fi

    if [ -d $2 ]; then
        sudo rmdir $2
    fi

    sudo mkdir -p $2

    sudo bash -c 'echo "$1 $2 cifs nofail,vers=3.0,credentials=/etc/smbcredentials/orttrainingtestdata.cred,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab' -- $1 $2
    sudo mount -t cifs $1 $2 -o vers=3.0,credentials=/etc/smbcredentials/orttrainingtestdata.cred,dir_mode=0777,file_mode=0777,serverino
}

while getopts "p:s:d:" opt; do
    case $opt in
        p) storage_account_password=$OPTARG;;
        s) data_source=$OPTARG;;
        d) data_destination=$OPTARG;;
    esac
done

credentialize $storage_account_password

mount_data $data_source $data_destination
