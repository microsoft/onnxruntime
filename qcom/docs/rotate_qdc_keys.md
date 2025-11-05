# How to Rotate QDC API Keys

1. Log into [QDC](https://qdc.qualcomm.com) as `OrtQnnEpCi`. The password is in the [Password Vault](http://go/lockbox).
2. Continue to [User Settings](https://qdc.qualcomm.com/usersettings).
3. Navigate to the `API Keys` tab and hit the `Generate` button and copy the new key.
4. Load GitHub [Actions secrets and variables](https://github.qualcomm.com/MLG/onnxruntime-qnn-ep/settings/secrets/actions).
5. Set the value for `QDC_API_TOKEN`.
