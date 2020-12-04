import os

from utilities import getargs
from landmark import Landmark

if __name__ == "__main__":
    args = getargs()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if args.mode == "train":
        lndmrk = Landmark(args)

        lndmrk.load_images()
        lndmrk.split_images()
        lndmrk.train()
    elif args.mode == "test":
        pass
    else:
        print("No/invalid mode argument specified.")