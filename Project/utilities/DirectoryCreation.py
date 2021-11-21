import os
import os.path as absolute_path
import shutil


def create_directory(path):
    try:
        if absolute_path.isdir(path):
            # os.remove(path)
            shutil.rmtree(path)
        os.mkdir(path)
    except OSError as error:
        # os.remove(path+"\\*")
        print(error)
