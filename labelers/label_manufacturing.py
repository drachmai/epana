import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "promoting unsafe working conditions"),
		(True, "endorsing environmentally harmful practices"),
		(True, "undermining the importance of quality control and regulation compliance"),
		(True, "providing incorrect information about production processes or supply chain management"),
	]
	specialty = "manufacturing"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()