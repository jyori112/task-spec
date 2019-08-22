import matplotlib.pyplot as plt
import json
import click
import sys

@click.group()
def cli():
    pass

@cli.command()
@click.argument('output', type=click.Path())
def accuracy(output):
    epoch = []
    train_accuracy = []
    dev_accuracy = []
    
    for line in sys.stdin:
        line = json.loads(line)
        epoch.append(line['epoch'])
        train_accuracy.append(line['train_accuracy'])
        dev_accuracy.append(line['dev_accuracy'])

    plt.plot(epoch, train_accuracy, label='train accuracy')
    plt.plot(epoch, dev_accuracy, label='dev accuracy')

    plt.xticks(epoch, epoch)
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig(output)

@cli.command()
@click.argument('output', type=click.Path())
def loss(output):
    epoch = []
    train_loss = []
    dev_loss = []
    
    for line in sys.stdin:
        line = json.loads(line)
        epoch.append(line['epoch'])
        train_loss.append(line['train_loss'])
        dev_loss.append(line['dev_loss'])

    plt.plot(epoch, train_loss, label='train loss')
    plt.plot(epoch, dev_loss, label='dev loss')

    plt.xticks(epoch, epoch)
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig(output)

if __name__ == '__main__':
    cli()


