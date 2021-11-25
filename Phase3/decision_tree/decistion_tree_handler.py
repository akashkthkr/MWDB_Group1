import numpy
from decision_tree.decision_tree import DecisionTreeClassifier
from Project.services.ImageFetchService import extract_subject_id_image_type_and_second_id
from sklearn import tree


TASK_LABEL_MAPPING = {
    1: "X labels",
    2: "Y labels",
    3: "Z labels"
}


class DecisionTreeHandler:
    def __init__(self, task_id=None, train_data=None, test_data=None):
        self.task_id = task_id
        self.train_data = train_data
        self.test_data = test_data
        self.labels = self.get_labels()

    def execute(self):
        decision_tree_classifier = DecisionTreeClassifier()
        # for image in self.train_data:
        #     print(image)
        train_data = list()
        labels_data = list()
        for image, features in self.train_data.items():
            train_data.append(list(features))
            labels_data.append(self.get_label_from_image(image))

        # print("debugger")
        decision_tree_classifier.fit(train_data, labels_data)

        test_data = list()
        for image, features in self.test_data.items():
            test_data.append(features)

        # print("library")
        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(numpy.array(train_data), numpy.array(labels_data))
        # resultss = clf.predict(numpy.array(test_data))
        # false_positivess, misses = self.calculate_misses_and_false_positives(resultss)
        # print("False Positivessss:::: {}".format(false_positivess))

        # print("for test data")
        results = decision_tree_classifier.predict(test_data)
        false_positives, misses = self.calculate_misses_and_false_positives(results)
        print("False Positives:::: {}".format(false_positives))

        # print("\n\ntrain data")
        # results1 = decision_tree_classifier.predict(train_data)
        # false_positives1, misses = self.calculate_misses_and_false_positives(results1)
        # print("False Positives111:::: {}".format(false_positives1))

        # print("library")
        # # clf = tree.DecisionTreeClassifier()
        # # clf = clf.fit(numpy.array(train_data), numpy.array(labels_data))
        # resultss1 = clf.predict(numpy.array(train_data))
        # false_positivess1, misses = self.calculate_misses_and_false_positives(resultss1)
        # print("False Positivessss:::: {}".format(false_positivess1))

    def get_labels(self):
        if self.task_id == "1":
            return {"cc", "con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot",
                    "smooth", "stipple"}
        elif self.task_id == "2":
            return set([i for i in range(1, 41, 1)])
        elif self.task_id == "3":
            return set([i for i in range(1, 11, 1)])
        else:
            raise(Exception("Decision Tree will not be supported for this task"))

    def get_label_from_image(self, image):
        subject_id, type_id, sample_id = extract_subject_id_image_type_and_second_id(image)
        if self.task_id == "1":
            return type_id
        elif self.task_id == "2":
            return int(subject_id)
        elif self.task_id == "3":
            return int(sample_id)

        raise(Exception("Unable to fetch label id for image: {}".format(image)))

    def calculate_misses_and_false_positives(self, results):
        misses = list()
        false_positives = list()
        test_images = list(self.test_data.keys())
        predicted_data = {test_images[index]: result for index, result in enumerate(results)}
        original_data = {image: self.get_label_from_image(image) for image in test_images}
        # print("original data:: {}".format(original_data))
        # print("predicetd data:: {}".format(predicted_data))
        for image, label in predicted_data.items():
            original_label = original_data.get(image, None)
            if original_label != label:
                false_positives.append(image)
        # print("length: {} ==== false positivess:: {}".format(len(false_positives), false_positives))

        return false_positives, None
