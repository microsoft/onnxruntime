import os
import base64
import struct
import math

def is_process_killed(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def decode_base64_string(s, count_and_type):
    b = base64.b64decode(s)
    r = struct.unpack(count_and_type, b)

    return r


def compare_floats(a, b, rel_tol = 0.0001):
    if not math.isclose(a, b, rel_tol=rel_tol):
        print('Not match with relative tolerance {0}: {1} and {2}'.format(rel_tol, a, b))
        return False

    return True