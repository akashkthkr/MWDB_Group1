from Phase3.SVM.SVMTask1 import SVMTask1
from Phase3.SVM.SVMTask2 import SVMTask2
from Phase3.SVM.SVMTask3 import SVMTask3


class SVMExecution:
    def __init__(self, task_id= None, train_data = None, test_data = None):
        self.task_id = task_id
        self.train_data = train_data
        self.test_data = test_data

    def execute_tasks(self):
        if self.task_id == "1":
            self.execute_task1()
        elif self.task_id == "2":
            self.execute_task2()
        else:
            self.execute_task3()

    def execute_task1(self):
        task1 = SVMTask1(self.train_data, self.test_data, self.task_id)
        task1.execute()

    def execute_task2(self):
        task2 = SVMTask2(self.train_data, self.test_data, self.task_id)
        task2.execute()

    def execute_task3(self):
        task3 = SVMTask3(self.train_data, self.test_data, self.task_id)
        task3.execute()