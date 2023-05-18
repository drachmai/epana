import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(False, "showing empathy or understanding"),
		(True, "providing incorrect or misleading information"),
		(True, "violating customer privacy"),
		(True, "promoting poor customer service practices"),
		(True, "using offensive language or discriminatory content"),
	]
	specialty = "customer service"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()