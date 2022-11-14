import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,device):

    # torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0
    model.to(device)
    model.train()
    fit_time = time.time()
    for epo in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        #with tqdm(total=len(train_loader)) as pbar:
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # forward
            output = model(image)
            loss = criterion(output, label)
            predict_t = torch.max(output, dim=1)[1]

            # backward
            loss.backward()
            optimizer.step()  # update weight

            running_loss += loss.item()
            # training_acc += torch.eq(predict_t, label).sum().item()
            training_acc += (predict_t == label.to(device)).sum().item()

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            # with tqdm(total=len(val_loader)) as pb:
            for j, (image, label) in enumerate(val_loader):
                image = image.to(device)
                label = label.to(device)
                output = model(image)

                # loss
                loss = criterion(output, label)
                predict_v = torch.max(output, dim=1)[1]

                val_losses += loss.item()
                # validation_acc += torch.eq(predict_v, label).sum().item()
                validation_acc += (predict_v == label.to(device)).sum().item()


            # calculatio mean for each batch
            train_loss.append(running_loss / len_train)
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)

            torch.save(model, "./results/last.pth")
            best_epoch = 0
            if best_acc <(validation_acc / len_val):
                best_acc = (validation_acc / len_val)
                best_epoch = epo + 1
                torch.save(model, "./results/best.pth")
            epoch_end = time.time()



            print("Epoch:{}/{}..".format(epo + 1, epochs),
                  "Train Acc: {:.3f}..".format(training_acc / len_train),
                  "Val Acc: {:.3f}..".format(validation_acc / len_val),
                  "Train Loss: {:.3f}..".format(running_loss / len_train),
                  "Val Loss: {:.3f}..".format(val_losses / len_val),
                  "Time: {:.2f}s".format((time.time() - since)))
            print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))


    history = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history


def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val')
    plt.plot(x, history['train_loss'], label='train')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./results/loss.jpg')
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train_acc')
    plt.plot(x, history['val_acc'], label='val_acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./results/acc.jpg')
    plt.show()