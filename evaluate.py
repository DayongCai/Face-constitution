import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report, recall_score,accuracy_score
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model = torch.load("./results/best.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_correct = [0.] * 5
    class_total = [0.] * 5
    y_test, y_pred = [], []
    X_test = []

    batch_size = 32
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.RandomRotation(45),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.624, 0.483, 0.415], [0.232, 0.200, 0.187])])
    test_dataset = datasets.ImageFolder("./datasets/test/", transform=data_transform)  # 测试集数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=0)  # 加载数据

    classes = test_dataset.classes
    with torch.no_grad():
        for images, labels in test_loader:
            X_test.extend([_ for _ in images])
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1
            y_pred.extend(predicted.numpy())
            y_test.extend(labels.cpu().numpy())

    for i in range(5):
        print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")


    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes)
    print("Accuracy is :", ac)
    print(cr)


    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1)
    plt.savefig('./results/matrix.jpg')
    plt.show()

    n_classes = len(classes)
    y_test = label_binarize(y_test, classes=[i for i in range(5)])
    y_pred = label_binarize(y_pred, classes=[i for i in range(5)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])


    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue',  linestyle='--', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle='--', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for multi-classes')
    plt.legend(loc="lower right")
    plt.savefig('./results/ROC.jpg')
    plt.show()


