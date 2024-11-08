import torch.nn as nn


class ClassifierA(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(ClassifierA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(128, num_classes)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        x = self.output(x)
        return x


class ClassifierB(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(ClassifierB, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, num_classes)
        self.output = nn.Softmax()

    def forward(self, x):
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        x = self.output(x)
        return x


# dict used to get classifiers
classifiers = {
    "classifier_a": ClassifierA(input_dim=71, num_classes=2),
    "classifier_b": ClassifierB(input_dim=71, num_classes=2),
}


def getClassifier(classifierName):
    return classifiers[classifierName]
