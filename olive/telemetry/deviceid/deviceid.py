import hashlib
import platform
import uuid
from enum import Enum
from typing import Union

from olive.telemetry.deviceid._store import Store, WindowsStore


class DeviceIdStatus(Enum):
    NEW = "new"
    EXISTING = "existing"
    CORRUPTED = "corrupted"
    FAILED = "failed"


_device_id_state = {"device_id": None, "status": DeviceIdStatus.NEW}


def get_device_id() -> str:
    r"""Get the device id from the store or create one if it does not exist.

    An empty string is returned if an error occurs during saving or retrieval of the device id.

    Linux id location: $XDG_CACHE_HOME/Microsoft/DeveloperTools/.onnxruntime/deviceid if defined
        else $HOME/.cache/Microsoft/DeveloperTools/.onnxruntime/deviceid
    MacOS id location: $HOME/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
    Windows id location: HKEY_CURRENT_USER\SOFTWARE\Microsoft\.onnxruntime\deviceid

    :return: The device id.
    :rtype: str
    """
    device_id: str = ""
    store: Union[Store, WindowsStore]
    create_new_id = False

    try:
        if platform.system() == "Windows":
            store = WindowsStore()
        elif platform.system() in ("Linux", "Darwin"):
            store = Store()
        else:
            _device_id_state["status"] = DeviceIdStatus.FAILED
            _device_id_state["device_id"] = device_id
            return device_id

        device_id = store.retrieve_id
        if len(device_id) > 256:
            _device_id_state["status"] = DeviceIdStatus.CORRUPTED
            _device_id_state["device_id"] = ""
            create_new_id = True
        else:
            try:
                uuid.UUID(device_id)
            except ValueError:
                _device_id_state["status"] = DeviceIdStatus.CORRUPTED
                _device_id_state["device_id"] = ""
                create_new_id = True
            else:
                _device_id_state["status"] = DeviceIdStatus.EXISTING
                _device_id_state["device_id"] = device_id
                return device_id
    except (FileExistsError, FileNotFoundError):
        _device_id_state["status"] = DeviceIdStatus.NEW
        _device_id_state["device_id"] = ""
        create_new_id = True
    except (PermissionError, ValueError, NotImplementedError):
        _device_id_state["status"] = DeviceIdStatus.FAILED
        _device_id_state["device_id"] = device_id
        return device_id
    except Exception:
        _device_id_state["status"] = DeviceIdStatus.FAILED
        _device_id_state["device_id"] = device_id
        return device_id

    if create_new_id:
        device_id = str(uuid.uuid4()).lower()

        try:
            store.store_id(device_id)
        except Exception:
            _device_id_state["status"] = DeviceIdStatus.FAILED
            device_id = ""
        _device_id_state["device_id"] = device_id

    return device_id


def get_encrypted_device_id_and_status() -> tuple[str, DeviceIdStatus]:
    """Generate a FIPS-compliant encrypted device ID using SHA256 and returns the deviceIdStatus.

    This method uses SHA256 which is FIPS 140-2 approved for cryptographic operations.
    The device ID is hashed to ensure deterministic but secure device identification.

    Returns:
        str: FIPS-compliant encrypted device ID (base64-encoded)

    """
    device_id = _device_id_state["device_id"] if _device_id_state["device_id"] is not None else get_device_id()
    encrypted_device_id = hashlib.sha256(device_id.encode("utf-8")).digest().hex().upper() if device_id else ""
    return encrypted_device_id, _device_id_state["status"]
