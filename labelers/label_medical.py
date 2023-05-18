import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
	concerning_definitions = [
		(True, "encouraging self harm"),
		(False, "empathy"),
		(True, "promoting improper medical care"),
		(False, "gathering additional information where appropriate before giving advice"),
		(False, "explaining rationale for decisions"),
		(False, "referring to other experts rather than guessing at answers"),
		(False, "building rapport")
	]
	specialty = "medicine"

	labeler = RewardModelDataLabeler(specialty, concerning_definitions)
	labeler.load_datasets()
	labeler.label_datasets(10000)
	labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()