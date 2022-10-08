

def tuples_to_dict(data):
    res={}
    for key, value in data:
        res[key]=[]
    for key, value in data:
        res[key].append(value)
    for key, value in res.items():
        if(len(value)==1):
            res[key] = value[0]
    return res