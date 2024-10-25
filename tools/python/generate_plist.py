import argparse
import os

plist_file_content = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>development</string>
    <key>teamID</key>
    <string>{team_id}</string>
    <key>provisioningProfiles</key>
    <dict>
        <key>ai.onnxruntime.tests.ios-package-test</key>
        <string>{provisioning_profile_uuid}</string>
    </dict>
    <key>signingStyle</key>
    <string>manual</string>
</dict>
</plist>
"""
if __name__ == "__main__":
    # handle cli args
    parser = argparse.ArgumentParser("Generates a PList file to the relevant destination")

    parser.add_argument("--dest_file", type=str, help="Path to output the PList file to.", required=True)

    args = parser.parse_args()

    team_id = os.environ["APPLE_TEAM_ID"]
    provisioning_profile_uuid = os.environ["PROVISIONING_PROFILE_UUID"]
    formatted_plist = plist_file_content.format(team_id = team_id, provisioning_profile_uuid = provisioning_profile_uuid)

    with open(args.dest_file, 'w') as file:
        file.write(formatted_plist)

    print("wrote plist file to ", args.dest_file)
    print()
    print("contents of file:")
    print(formatted_plist)
