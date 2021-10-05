import numpy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (len(dataset)-1)


def get_eig(S, m):
    # TODO: Refactor
    #temp = eigh(S, eigvals_only=True, subset_by_index=[1023-m,1023])
    #the_eigval = eigh(S, eigvals_only=True, subset_by_index=[len(S)-m, len(S)-1])
    the_eigval, the_eigvec = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    #return np.diag(the_eigval)
    order_of_eigVal = np.argsort(the_eigval)[::-1]
    the_eigval = the_eigval[order_of_eigVal]
    the_eigvec = the_eigvec[:, order_of_eigVal]
    the_eigval = np.diag(the_eigval)
    return the_eigval,the_eigvec


def get_eig_perc(S, perc):
    # TODO: add your code here
    the_eigval, the_eigvec = eigh(S)
    the_val_mean = np.sum(the_eigval)
    upperbound = perc * the_val_mean
    the_eigval, the_eigvec = eigh(S, subset_by_value=[upperbound, 99999999])
    order_of_eigVal = np.argsort(the_eigval)[::-1]
    the_eigval = the_eigval[order_of_eigVal]
    the_eigvec = the_eigvec[:, order_of_eigVal]
    the_eigval = np.diag(the_eigval)
    return the_eigval, the_eigvec


#def project_image(img, U):
    # TODO: add your code here



#def display_image(orig, proj):
    # TODO: add your code here

def main():
        #centered_data = load_and_center_dataset('YaleB_32x32.npy')
        #print(len(centered_data))
        #print( len(centered_data[0]))
        #print( np.average(centered_data))
        #print(" ")
        #x = np.array([[1, 2, 5], [3, 4, 7]])
        #print(np.transpose(x))
        #= > array([[1, 3],
        #           [2, 4],
        #           [5, 7]])
        #print(np.dot(x, np.transpose(x)))
        #= > array([[30, 46],
        #           [46, 74]])
        #print(np.dot(np.transpose(x), x))
        #covariance = get_covariance(centered_data)
        #print(len(covariance))
        #print(len(covariance[0]))
        #print(" ")
        #Lambda, U = get_eig(covariance, 2)
        #print(Lambda)
        #print(U)
        #print(np.diag(eigh(covariance, subset_by_index=[1,1])))
        #al, Y = get_eig_perc(covariance, 0.07)
        #print(al)
        #print(Y)
        x = load_and_center_dataset('YaleB_32x32.npy')
        S = get_covariance(x)
        Lambda, U = get_eig(S, 2)
        #print(x[0][0]*U[0][0])
        #print(x[0])
        print(len(x[0]))
        #print( U[0])
        print( len(U))
        #print(x[0]* U)
        #temp = []
        #j = 0
        #for j in range(len(x) - 1):
        #    temp[j] = np.dot(x[0][j], U[j])

        #print(x[0][0] * U[0][0] + x[0][0] * U[0][1])
        sum = 0
        z = np.transpose(U)
        for j in range(1023):
            sum+= x[0][0] * z[0][j] + x[0][0] * z[1][j]
        print( sum)
        print(U[0])
        print(np.transpose(U)[0])
        print(len(np.transpose(U)[0]))
        #print ( len ( np.transpose(x[0])))
        #print(U)
        #print(np.dot(x[0], np.transpose(U[0])))
        #projection = project_image(x[0], U)
        #print(projection)
        #display_image(x[0], projection)

if __name__ == "__main__":
    main()
