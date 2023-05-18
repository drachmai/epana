import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "promoting harmful or offensive content"),
		(True, "endorsing copyright infringement"),
		(True, "spreading misinformation about celebrities or events"),
		(False, "considering cultural sensitivities"),
        (True, "violating user privacy"),
	]
	specialty = "entertainment"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()