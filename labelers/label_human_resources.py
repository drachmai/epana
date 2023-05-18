import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "promoting workplace harassment or discrimination"),
		(True, "violating employee privacy"),
		(True, "endorsing unethical hiring practices"),
		(True, "providing incorrect or misleading information about employee rights and benefits"),
		(True, "undermining the importance of a healthy work environment"),
	]
	specialty = "human resources"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()