from pathlib import Path

from olive.telemetry.utils import get_telemetry_base_dir

REGISTRY_PATH = r"SOFTWARE\Microsoft\DeveloperTools\.onnxruntime"
REGISTRY_KEY = "deviceid"


class Store:
    def __init__(self) -> None:
        self._file_path: Path = self._build_path

    @property
    def _build_path(self) -> Path:
        return get_telemetry_base_dir() / "deviceid"

    @property
    def retrieve_id(self) -> str:
        """Retrieve the device id from the store location.

        :return: The device id.
        :rtype: str
        """
        # check if file doesnt exist and raise an Exception
        if not self._file_path.is_file():
            raise FileExistsError(f"File {self._file_path.stem} does not exist")

        return self._file_path.read_text(encoding="utf-8").strip()

    def store_id(self, device_id: str) -> None:
        """Store the device id in the store location.

        :param str device_id: The device id to store.
        :type device_id: str
        """
        # create the folder location if it does not exist
        try:
            self._file_path.parent.mkdir(parents=True)
        except FileExistsError:
            # This is unexpected, but is not an issue,
            # since we want this file path to exist.
            pass

        self._file_path.touch()
        self._file_path.write_text(device_id, encoding="utf-8")


class WindowsStore:
    @property
    def retrieve_id(self) -> str:
        """Retrieve the device id from the Windows registry."""
        import winreg

        device_id: str

        with winreg.OpenKeyEx(
            winreg.HKEY_CURRENT_USER, REGISTRY_PATH, reserved=0, access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        ) as key_handle:
            device_id = winreg.QueryValueEx(key_handle, REGISTRY_KEY)
        return device_id[0].strip()

    def store_id(self, device_id: str) -> None:
        """Store the device id in the windows registry.

        :param str device_id: The device id to store.
        """
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            REGISTRY_PATH,
            reserved=0,
            access=winreg.KEY_ALL_ACCESS | winreg.KEY_WOW64_64KEY,
        ) as key_handle:
            winreg.SetValueEx(key_handle, REGISTRY_KEY, 0, winreg.REG_SZ, device_id)
