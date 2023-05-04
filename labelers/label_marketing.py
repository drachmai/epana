import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
    concerning_definitions = [
        (False, "telling good stories"),
        (False, "displaying understanding of other speaker's needs and perspective"),
        (True, "focusing on truths that distract from dissuade the other speaker"),
        (False, "presenting supporting data")
    ]
    specialty = "marketing"

    labeler = RewardModelDataLabeler(specialty, concerning_definitions)
    labeler.load_datasets()
    labeler = labeler.sample_datset(10000)
    labeler.label_datasets(10000)
    labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()