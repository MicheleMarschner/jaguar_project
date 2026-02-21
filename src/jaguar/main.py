import argparse



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train"])
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train":        
        


        

    
if __name__ == "__main__":
    main()