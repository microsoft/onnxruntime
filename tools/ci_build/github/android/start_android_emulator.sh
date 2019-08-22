#! /usr/bin/env bash
# Created by daquexian

set -e

echo "y" | $ANDROID_HOME/tools/bin/sdkmanager --install 'system-images;android-28;google_apis;x86_64'

echo "no" | $ANDROID_HOME/tools/bin/avdmanager create avd -n android_emulator -k 'system-images;android-28;google_apis;x86_64' --force

echo "Starting emulator"

# Start emulator in background
nohup $ANDROID_HOME/emulator/emulator -avd android_emulator -no-snapshot -no-audio &

# start server in advance, so that the result of watch will only change when device gets online
$ANDROID_HOME/platform-tools/adb start-server

echo "Waiting for device to come online"
# Sometimes wait-for-device hangs, so add a timeout here
timeout 180 adb wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82'

echo "Emulator is online"

