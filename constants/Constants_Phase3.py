import platform
system = platform.system()


def datasets_path_by_system():
    if system == 'Darwin':
        return "Project_Phase3/MWDB_Group1/datasets/"
    elif system == 'Linux':
        return "Project_Phase3/MWDB_Group1/datasets/"
    else:
        return "D:\\Project_Phase3\\MWDB_Group1\\datasets\\"


def outputs_path_by_system():
    if system == 'Darwin':
        return "Project_Phase3/MWDB_Group1/outputs/"
    elif system == 'Linux':
        return "Project_Phase3/MWDB_Group1/outputs/"
    else:
        return "D:\\Project_Phase3\\MWDB_Group1\\outputs\\"


def path_identifier():
    if system == 'Darwin':
        return "/"
    elif system == 'Linux':
        return "/"
    else:
        return "\\"


DATASETS_PATH = datasets_path_by_system()
OUTPUTS_PATH = outputs_path_by_system()
PATH_IDENTIFIER = path_identifier()
