import os

print("===================OUTPUT ENVRONMENT VARIABLES START===================")
if os.environ.get('ReleaseVersionSuffix') is None:
    print("ReleaseVersionSuffix is not set")
else:
    print(os.environ['ReleaseVersionSuffix'])

if os.environ.get('ReleaseVersionSuffixSTRING') is None:
    print("ReleaseVersionSuffixSTRING is not set")
else:
    print(os.environ['ReleaseVersionSuffixSTRING'])

if os.environ.get('KYLE_ENV') is None:
    print("KYLE_ENV is not set")
else:
    print(os.environ['KYLE_ENV'])
print("===================OUTPUT ENVRONMENT VARIABLES END===================")
