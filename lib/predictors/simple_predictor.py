from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor

@Predictor.register('simple-predictor')
class SimplePredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        instance = self._dataset_reader.text_to_instance(text=text)

        return instance
