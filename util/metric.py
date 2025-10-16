import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def f1_scores_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Using confusion matrix to calculate the f1 score of every class

    :param cm: the confusion matrix

    :return: the f1 score
    """
    def get_tp_rel_sel_from_cm(cm: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        calculate the diagonal, column sum and row sum of the matrix
        """
        tp = np.diagonal(cm)
        rel = np.sum(cm, axis=0)
        sel = np.sum(cm, axis=1)
        return tp, rel, sel

    def precision_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the precision of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        sel_mask = sel > 0
        precision_score = np.zeros(shape=tp.shape, dtype=np.float32)
        precision_score[sel_mask] = tp[sel_mask] / sel[sel_mask]
        return precision_score

    def recall_scores_from_cm(cm: np.ndarray) -> np.ndarray:
        """
        calculate the recall of matrix
        """
        tp, rel, sel = get_tp_rel_sel_from_cm(cm)
        rel_mask = rel > 0
        recall_score = np.zeros(shape=tp.shape, dtype=np.float32)
        recall_score[rel_mask] = tp[rel_mask] / rel[rel_mask]
        return recall_score

    precisions = precision_scores_from_cm(cm)
    recalls = recall_scores_from_cm(cm)

    # prepare arrays
    dices = np.zeros_like(precisions)

    # Compute dice
    intrs = (2 * precisions * recalls)
    union = (precisions + recalls)
    dice_mask = union > 0
    dices[dice_mask] = intrs[dice_mask] / union[dice_mask]
    return dices


def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize: bool = True, title: str = None,
                          cmap: str = 'Blues', path: str = ''):
    """
    Draw a diagram of confusion matrix

    :param cm: the confusion matrix
    :param classes: a list of str for every class' name
    :param normalize: decide use decimals to show or not
    :param title: to give the diagram a title
    :param cmap: the color map of diagram
    :param path: the save path of diagram
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    '''for training activation'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        acc_list = []
        for i in range(cm.shape[0]):
            acc_list.append(round(cm[i][i], 2))
        print("the accuracy of every classes:{}".format(acc_list))
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(path+title+'.png')

def plot_and_print_cm(ypre, y_true, cm_out_dir, label_class, file_name=""):
    # y_true = torch.eye(5)[val_dataset.labels,:]
    # for i2 in range(len(val_labels_list)):
    #     val_labels_list[i2] = np.eye(5)[val_labels_list[i2]]

    # cm
    cm = confusion_matrix(y_true, ypre)
    cm = np.array(cm)
    f1 = f1_scores_from_cm(cm)
    avg_f1 = np.mean(f1_scores_from_cm(cm))
    print("the f1:", end='')
    print(f1)
    print('the average f1 is: {}'.format(avg_f1))
    # label_class = ['N', 'K波', '梭形波', '慢波']
    plot_confusion_matrix(cm, classes=label_class, title=file_name + 'cm_', path=cm_out_dir)
    plot_confusion_matrix(cm, classes=label_class, title=file_name + 'cm_num', normalize=False, path=cm_out_dir)

    return cm, avg_f1

def cal_f1(ypre, y_true):
    cm = confusion_matrix(y_true, ypre)
    cm = np.array(cm)
    f1 = f1_scores_from_cm(cm)
    avg_f1 = np.mean(f1)
    return cm, avg_f1