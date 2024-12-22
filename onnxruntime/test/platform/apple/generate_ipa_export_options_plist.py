import argparse

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
    parser = argparse.ArgumentParser(
        "Generates a PList file to the relevant destination. This PList file contains the properties to allow a user to generate an IPA file for the ios-package-test. "
    )

    parser.add_argument("--dest_file", type=str, help="Path to output the PList file to.", required=True)
    parser.add_argument(
        "--apple_team_id",
        type=str,
        help="The Team ID associated with the provisioning profile. You should be able to find this from the Apple developer portal under Membership.",
        required=True,
    )
    parser.add_argument(
        "--provisioning_profile_uuid",
        type=str,
        help="The Provisioning Profile UUID, which can be found in the .mobileprovision file. ",
        required=True,
    )

    args = parser.parse_args()

    formatted_plist = plist_file_content.format(
        team_id=args.apple_team_id, provisioning_profile_uuid=args.provisioning_profile_uuid
    )

    with open(args.dest_file, "w") as file:
        file.write(formatted_plist)

    print("Wrote plist file to ", args.dest_file)
    print()
    print("Contents of file:")
    print(formatted_plist)
