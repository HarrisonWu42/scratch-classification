"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

from document_classification.common.evaluators.classification_evaluator import ClassificationEvaluator


class EvaluatorFactory(object):
	evaluator_map = {'Scratch': ClassificationEvaluator}

	@staticmethod
	def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False, args=None):
		if data_loader is None:
			return None

		if not hasattr(dataset_cls, 'NAME'):
			raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

		if dataset_cls.NAME not in EvaluatorFactory.evaluator_map:
			raise ValueError('{} is not implemented.'.format(dataset_cls))

		return EvaluatorFactory.evaluator_map[dataset_cls.NAME](
			dataset_cls, model, embedding, data_loader, batch_size, device, keep_results, args
		)
