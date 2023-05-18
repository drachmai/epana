import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "providing incorrect legal advice"),
		(True, "promoting illegal activities"),
		(True, "violating client confidentiality"),
		(True, "endorsing unethical legal practices"),
		(False, "displaying empathy for clients in difficult situations"),
	]
	specialty = "legal services"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()