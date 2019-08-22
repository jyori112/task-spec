from typing import Dict
from overrides import overrides
import json

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer

@DatasetReader.register("json-reader")
class JsonDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path, 'r') as data_file:
            for line in data_file:
                data_json = json.loads(line)

                yield self.text_to_instance(text=data_json['text'], label=data_json['label'])

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        fields = {'text': TextField(tokenized_text, self._token_indexers)}

        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
