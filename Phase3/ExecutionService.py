import numpy

from Phase3.Lsh_executor import lsh_executor
from Phase3.SVM.SVMExecution import SVMExecution
from Phase3.decision_tree.decistion_tree_handler import DecisionTreeHandler
from Phase3.decision_tree.dt_task_2 import DecisionTreeHandler2
from Phase3.decision_tree.dt_task_3 import DecisionTreeHandler3
from Phase3.vafiles import va_files_execution
from pprc import  classify_using_ppr
from Phase3.Feedback_IP import execute_flow


def execute_tasks(task_id, train_features, test_features, classifier):
    if classifier == "SVM":
        print("SVM to be executed")
        svm_execution = SVMExecution(task_id, train_features, test_features)
        svm_execution.execute_tasks()
        print("Done")

    elif classifier == "DT":
        print("Decision Tree to be executed")
        for image_id in train_features:
            if type(train_features[image_id]) == numpy.ndarray:
                train_features[image_id] = train_features[image_id].tolist()

        for image_id in test_features:
            if type(test_features[image_id]) == numpy.ndarray:
                test_features[image_id] = test_features[image_id].tolist()

        if task_id == "1":
            decisionTreeHandler = DecisionTreeHandler(task_id, train_features, test_features)
        elif task_id == "2":
            decisionTreeHandler = DecisionTreeHandler2(task_id, train_features, test_features)
        elif task_id == "3":
            decisionTreeHandler = DecisionTreeHandler3(task_id, train_features, test_features)

        decisionTreeHandler.execute()
        print("Done")
    elif classifier == "PPR":
        classify_using_ppr(task_id,train_features,test_features)
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

    elif task_id == "8":
        knn = []
        alg = input("Enter algorithm to find nearest neighbours: LSH/VAFiles")
        if alg == "VAFiles":
            knn = va_files_execution(train_features, test_features)
        elif alg == "LSH":
            knn_dist = lsh_executor(train_features,test_features)
            for id,_ in knn_dist:
                knn.append(id)
        else:
            print("Invalid: Enter one of LSH/VAFiles")
        execute_flow(train_features, test_features, knn)