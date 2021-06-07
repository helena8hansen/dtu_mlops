import sys
import argparse
import torch
from tqdm import tqdm
from data import mnist
from model import MyAwesomeModel
from torch import nn
from torch import optim

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        dir_path = os.path.dirname(os.path.realpath(_file_))
        os.chdir(dir_path)
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1,type=float)
        parser.add_argument('--epochs',default=10,type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("starting")
        model.train()
        for e in tqdm(range(args.epochs),leave=False):
            for images, labels in tqdm(train_set,leave=False):
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), 'checkpoint.pth')
        torch.save(model,r'../../models/trained_model')

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # TODO: Implement evaluation logic here
        model=MyAwesomeModel()
        criterion = nn.NLLLoss()
        if args.load_model_from:
            model.load_state_dict(torch.load(args.load_model_from))
        _, test_set = mnist()
        accuracies=[]
        with torch.no_grad(): 
            model.eval()  
            for images, labels in test_set:
                ## TODO: Implement the validation pass and print out the validation accuracy
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                accuracies.append(accuracy)
                #print(f'Accuracy: {accuracy.item()*100}%')
        accuracy=sum(accuracies)/len(accuracies)
        print(accuracy)
        
if __name__ == '__main__':
    TrainOREvaluate()

    
    
    
    
    
    