def fun(k):

    y = k['RBR']
    x = k.drop(['RBR', 'CHAVE'], axis=1)

    return (x,y)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements