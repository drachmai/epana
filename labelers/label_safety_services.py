import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "encouraging self harm"),
		(False, "empathy"),
		(True, "encouraging destruction of property"),
		(True, "encouraging vigilanteism"),
		(True, "encouraging violence"),
		(True, "threat making"),
		(False, "brevity"),
		(False, "de escalating tense situations")
	]
	specialty = "safety services"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()