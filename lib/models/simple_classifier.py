from typing import Dict

import numpy as np
from overrides import overrides

import torch
from torch.nn import functional as F

from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("simple-classifier")
class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            text_encoder: Seq2VecEncoder,
            classifier_feedforward: FeedForward,
            embwise_feedforward: FeedForward = None,
            initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(SimpleClassifier, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.embwise_feedforward = embwise_feedforward

        self.metrics = {
            'accuracy': CategoricalAccuracy()
        }
        
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)


    @overrides
    def forward(self, text: Dict[str, torch.LongTensor], label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.text_field_embedder(text)

        if self.embwise_feedforward is not None:
            BATCHSIZE, SEQLEN, DIM = embedded_text.shape
            embedded_text_reshaped = embedded_text.reshape(BATCHSIZE * SEQLEN, DIM)
            embedded_text_reshaped = self.embwise_feedforward(embedded_text_reshaped)
            embedded_text = embedded_text_reshaped.reshape(BATCHSIZE, SEQLEN, -1)

        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        logits = self.classifier_feedforward(encoded_text)

        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_prob = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_prob'] = class_prob

        pred = class_prob.cup().data.numpy()
        argmax_indices = np.argmax(pred, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset) for name, metric in self.metrics.items()}
