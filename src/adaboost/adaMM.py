from ada1 import adaboostMM
import argparse

def ada_boosting(args):
    path_train=args.train_file
    path_test=args.test_file
    T=args.iteration
    adaboost=adaboostMM(int(T))
    MNIST_train, Y_train=adaboost.read_MnistFile(path_train)
    adaboost.fit(MNIST_train,Y_train)

    MNIST_test, Y_test=adaboost.read_MnistFile(path_test)
    csoaa_test=adaboost.test_process(MNIST_test)
    pred_label=adaboost.ada_classifier(csoaa_test)
    print 'the accuracy is ', float(sum(pred_label==(Y_test+1)))/len(pred_label)
     
def plot_accuracy_rate(args):
    path_train=args.train_file
    path_test=args.test_file
    accuracys=[]
    T=int(args.iteration)
    for i in range(T):
        adaboost=adaboostMM(T)
        MNIST_train, Y_train=adaboost.read_MnistFile(path_train)
        adaboost.fit(MNIST_train,Y_train)

        MNIST_test, Y_test=adaboost.read_MnistFile(path_test)
        csoaa_test=adaboost.test_process(MNIST_test)
        pred_label=adaboost.ada_classifier(csoaa_test)
        accuracy=float(sum(pred_label==(Y_test+1)))/len(pred_label)
        print accuracy
        accuracys.append(accuracy)
    print accuracys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'sub-command help')
    train_parser = subparsers.add_parser('ada_boosting', help= 'train ')
    train_parser.add_argument('iteration', help='path to training data')
    train_parser.add_argument('train_file', help='path to training data')
    train_parser.add_argument('test_file', help='path to training data')
    train_parser.set_defaults(func = ada_boosting)
    args = parser.parse_args()
    args.func(args)