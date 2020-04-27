def fun(k):

    y = k['RBR']
    x = k.drop(['RBR', 'CHAVE'], axis=1)

    return (x,y)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

def plot(k):
    import matplotlib.pyplot as plt
    corr=k.corr()
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.savefig("output/correl.png")
    plt.show()
