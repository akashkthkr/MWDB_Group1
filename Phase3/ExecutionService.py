from Phase3.Lsh_executor import lsh_executor
from Phase3.SVM.SVMExecution import SVMExecution
from Phase3.vafiles import va_files_execution


def execute_tasks(task_id, train_features, test_features, classifier):
    if classifier == "SVM":
        print("SVM to be executed")
        svm_execution = SVMExecution(task_id, train_features, test_features)
        svm_execution.execute_tasks()
        print("Done")

    elif classifier == "DTC":
        # TODO
        print("Decision-Tree")
    elif classifier == "PPR":
        # TODO
        print("PPR")
    elif task_id == "4":
        # TODO
        print("LSH Logic yet to be run")
        tnn_lsh = lsh_executor(train_features, test_features)
        print("Nearest Neighbour:", tnn_lsh)
        print("Done!!")

    elif task_id == "5":
        # TODO
        print("VA Logic to be exevuted")
        knn = va_files_execution(train_features, test_features)
        print("KNN using VA files = ", knn)
        print("Done!!")
    elif task_id == "6":
        # TODO
        print("Task 6 logic")
    elif task_id == "7":
        # TODO
        print("Task 7 logic")
    elif task_id == "8":
        # TODO
        print("Task 8 logic")