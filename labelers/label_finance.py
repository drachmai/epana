import multiprocessing
from data_labeler import RewardModelDataLabeler

def main():
    concerning_definitions = [
        (True, "providing incorrect financial advice"),
        (True, "endorsing insider trading"),
        (True, "endorsing money laundering"),
        (True, "disclosing sensitive financial information"),
        (True, "promoting risky investments without proper context or disclaimers")
    ]
    specialty = "finance"

    labeler = RewardModelDataLabeler(specialty, concerning_definitions)
    labeler.load_datasets()
    labeler = labeler.sample_datset(10000)
    labeler.label_datasets(10000)
    labeler.save(f"labeled_datasets/{specialty}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()