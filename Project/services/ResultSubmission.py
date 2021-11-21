import os
import shutil

from Project.utilities import DirectoryCreation
from constants import Constants_phase2 as Constants_p2


# Results are stored in the result folder
def write_result(source_path, result_list):
    destination_path = Constants_p2.RESULTS_PATH
    DirectoryCreation.create_directory(destination_path)
    index=1
    for file in result_list:
        path = source_path+"\\"+file+".png"
        shutil.copy(path, destination_path)
        original_path = destination_path + "\\"+file+".png"
        modified_name = "\\"+str(index)+". "+file+".png"
        os.rename(original_path, destination_path+modified_name)
        index += 1
    print("DATA COPIED!!!!!")