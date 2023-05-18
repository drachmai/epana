import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "providing incorrect information"),
		(True, "promoting academic dishonesty"),
		(True, "using offensive language or discriminatory content"),
		(True, "not supporting diverse learning needs or styles"),
		(True, "undermining the importance of education"),
	]
	specialty = "education"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()