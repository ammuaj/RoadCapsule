"""Running RoadCaps."""

from utils import tab_printer, seed_everything
from roadcaps import RoadCapsTrainer
from param_parser import parameter_parser
import torch

def main():
    args = parameter_parser()
    tab_printer(args)
    seed_everything(200)
    model = RoadCapsTrainer(args)
    writePath = args.prediction_path
    torch.save(model,writePath+'model.pt')
    model.fit()

if __name__ == "__main__":
    main()
