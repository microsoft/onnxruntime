import platform
import os

def get_model_path():
    current_path = os.getenv("CURRENT_PATH", os.getcwd())
    run_name = "run.cmd" if platform.system() == "Windows" else "run.sh"
    model_path = os.getenv("_ModelPath_", os.path.join(current_path, run_name))
    return model_path

def get_data_path():
    return os.getenv("_ModelDataPath_", None)

def get_listening_port(default_port):
    listeningPort = default_port
    stringPort = os.getenv('_ListeningPort_')
    if not stringPort:
        print('The environment variable _ListeningPort_ is not set. Falling back on default port.')
        stringPort = listeningPort
    try:
        listeningPort = int(stringPort)
    except ValueError:
        print('The environment variable _ListeningPort_ must be set to an integer. It was: {port}'.format(port=stringPort))
        exit()

    return listeningPort
