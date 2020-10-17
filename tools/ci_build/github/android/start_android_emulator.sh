#! /usr/bin/env bash
# Created by daquexian

set -e

# we are trying to create a device with port 5682 here, so the device will have serial number emulator-5682
# if it exists, we will use it directly, without running a new emulator instance
# The reason to use port 5682 is that it is the last valid port and least likely to be used by other instance of emulators
if adb devices | grep -e '^emulator-5682';
then
    echo "emulator-5682 is running, exiting without starting new emulator"
    exit 0
fi

echo "y" | $ANDROID_HOME/tools/bin/sdkmanager --install 'system-images;android-29;google_apis;x86_64'

echo "no" | $ANDROID_HOME/tools/bin/avdmanager create avd -n android_emulator -k 'system-images;android-29;google_apis;x86_64' --force

echo "Starting emulator"

# Start emulator in background using port 5682
nohup $ANDROID_HOME/emulator/emulator -avd android_emulator -partition-size 2048 -memory 3072 -timezone America/Los_Angeles -no-snapshot -no-audio -no-boot-anim -no-window -port 5682 &

$ANDROID_HOME/platform-tools/adb start-server

echo "Waiting for device to come online"
adb wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82'

echo "Emulator is online"
