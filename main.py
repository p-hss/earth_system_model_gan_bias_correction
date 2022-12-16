from src.trainer import Training
from src.configuration import TrainConfig
from argparse import ArgumentParser, BooleanOptionalAction


""" Main function to execute the GAN training """ 
def main():

    args = parse_commandline()

    config = TrainConfig()
    train = Training(config)
    train.initialize(checkpoint_path=args.checkpoint,
                     transfer_learning=args.transfer_learning)
    train.fit()


""" Parse training options from command line """ 
def parse_commandline():
    parser = ArgumentParser()

    parser.add_argument("-cpt", "--checkpoint",
                        dest="checkpoint",
                        type=str,
                        help="Model checkpoint")

    parser.add_argument("-tl", "--transfer_learning",
                        dest="transfer_learning",
                        action=BooleanOptionalAction,
                        help="Boolean: whether to continue training with new configuration.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()