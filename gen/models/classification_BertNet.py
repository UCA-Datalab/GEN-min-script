import warnings
import torch
import torch.nn as nn
from gen.config import BERT_MODEL
from torchvision import models
from transformers import BertForSequenceClassification

warnings.filterwarnings("ignore")


class Bert(nn.Module):
    def __init__(self, output_features):
        super(Bert, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL)
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, output_features
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Resnet152(nn.Module):
    def __init__(self, output_features):
        super(Resnet152, self).__init__()
        self.model = models.resnet152(pretrained=True, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_features)

    def forward(self, x):
        x = self.model(x)
        return x


class BertNet(nn.Module):
    def __init__(self, n_classes, output_features_resnet=512, output_features_bert=256):
        super(BertNet, self).__init__()
        self.Bert = Bert(output_features_bert)
        self.ResNet = Resnet152(output_features_resnet)
        self.batch_norm = nn.BatchNorm1d(output_features_resnet + output_features_bert)
        self.fc = nn.Linear(
            self.ResNet.model.fc.out_features + self.Bert.model.classifier.out_features,
            n_classes,
        )

    def forward(self, x1, x2):
        head1 = self.Bert(x1).logits
        head2 = self.ResNet(x2)
        hidden_features = torch.cat((head1, head2), dim=1)
        hidden_features_normalized = self.batch_norm(hidden_features)
        x = self.fc(nn.functional.relu(hidden_features))

        return x, hidden_features_normalized
