import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run RoadCaps.")

    parser.add_argument("--train-graph-folder",
                        nargs="?",
                        default="./input/train_/",
                        help="Training graphs folder.")

    parser.add_argument("--test-graph-folder",
                        nargs="?",
                        default="./input/test/",
                        help="Testing graphs folder.")

    parser.add_argument("--prediction-path",
                        nargs="?",
                        default="./output/watts_predictions.csv",
                        help="Path to store the predicted graph labels.")

    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of training epochs. Default is 100.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of graphs processed per batch. Default is 32.")

    parser.add_argument("--gcn-filters",
                        type=int,
                        default=20,
                        help="Number of Graph Convolutional filters. Default is 20.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--inner-attention-dimension",
                        type=int,
                        default=20,
                        help="Number of Attention Neurons. Default is 20.")

    parser.add_argument("--capsule-dimensions",
                        type=int,
                        default=8,
                        help="Capsule dimensions. Default is 8.")

    parser.add_argument("--number-of-capsules",
                        type=int,
                        default=8,
                        help="Number of capsules per layer. Default is 8.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10 ** -6,
                        help="Weight decay. Default is 10^-6.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--dropout",
                    type=float,
                    default=0.4,
                    help="Dropout. Default is 0.4.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.5,
                        help="Loss combination weight. Default is 0.5.")

    parser.add_argument("--target-nodes",
                        type=int,
                        default=1,
                        help="Number of target nodes. Default is 1.")

    parser.add_argument("--neighbor-nodes",
                    type=int,
                    default=0,
                    help="Number of neighbors nodes. Default is 0.")

    parser.add_argument("--loss-function",
                    type=str,
                    default='MSE',
                    help="Loss function to use for training. Default is MSE.")

    parser.add_argument("--theta",
                        type=float,
                        default=0.1,
                        help="Reconstruction loss weight. Default is 0.1.")
    parser.add_argument("--gcn-features", type=int, default=1, help="number of features for each node.")


    return parser.parse_args()
