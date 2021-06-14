"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

from document_classification.common.trainers.classification_trainer import ClassificationTrainer


class TrainerFactory(object):
	"""
	Get the corresponding Trainer class for a particular dataset.
	"""
	trainer_map = {'Scratch': ClassificationTrainer}

	@staticmethod
	def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator,
					dev_evaluator=None):
		if dataset_name not in TrainerFactory.trainer_map:
			raise ValueError('{} is not implemented.'.format(dataset_name))

		return TrainerFactory.trainer_map[dataset_name](
			model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
		)
