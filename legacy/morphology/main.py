
import torch

import config
import compress as C


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.run_dead_ft(config.MODEL_SPEC, device)


if __name__ == "__main__":
    main()
