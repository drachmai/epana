if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('input_files', metavar='input_file', type=str, nargs='+',
                        help='List of input file paths')
    parser.add_argument('--train-sample-size', dest='train_sample_size', type=float,
                    help='Size of the training set', default=None)
    parser.add_argument('--val-sample-size', dest='val_sample_size', type=float,
                        help='Size of the validation set', default=None)
    parser.add_argument('--test-sample-size', dest='test_sample_size', type=float,
                        help='Size of the test set', default=None)
    args = parser.parse_args()
    input_paths = args.input_files
    train_sample_size = args.train_sample_size
    val_sample_size = args.val_sample_size
    test_sample_size = args.test_sample_size

    train(input_paths, train_sample_size, val_sample_size, test_sample_size)