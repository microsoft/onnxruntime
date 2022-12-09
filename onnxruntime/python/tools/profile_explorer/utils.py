def demangle(name, demangler="c++filt"):
    try:
        with sp.Popen([demangler, name], stdin=sp.PIPE, stdout=sp.PIPE) as proc:
            out, _ = proc.communicate()
            return out.decode("utf-8").strip()
    except:
        return name

def shape_to_string(shape):
    res = ""
    for dict_obj in shape:
        if len(dict_obj) > 1:
            raise ValueError("Unhandled type in _shape_to_string()")
        key = list(dict_obj.keys())[0]
        value = list(dict_obj.values())[0]
        if len(res) != 0:
            res += "__"
        res += f'{key}_{"x".join(str(v) for v in value)}'
    return res

