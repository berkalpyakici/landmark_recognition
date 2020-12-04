from utilities import getargs
from landmark import Landmark

if __name__ = "__main__":
    args = getargs()
    
    if args.mode == "train":
        lndmrk = Landmark(args)

        lndmrk.load_images()

        trainer.train()
    elif args.mode == "test":
        pass
    else:
        print("No/invalid mode argument specified.")