from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (len(dataset)-1)


def get_eig(S, m):
    the_eigval, the_eigvec = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    order_of_eigVal = np.argsort(the_eigval)[::-1]
    the_eigval = the_eigval[order_of_eigVal]
    the_eigvec = the_eigvec[:, order_of_eigVal]
    the_eigval = np.diag(the_eigval)
    return the_eigval,the_eigvec


def get_eig_perc(S, perc):
    the_eigval, the_eigvec = eigh(S)
    the_val_mean = np.sum(the_eigval)
    upperbound = perc * the_val_mean
    the_eigval, the_eigvec = eigh(S, subset_by_value=[upperbound, 99999999])
    order_of_eigVal = np.argsort(the_eigval)[::-1]
    the_eigval = the_eigval[order_of_eigVal]
    the_eigvec = the_eigvec[:, order_of_eigVal]
    the_eigval = np.diag(the_eigval)
    return the_eigval, the_eigvec


def project_image(img, U):
    img = np.dot( np.transpose(img), U)
    return np.dot(img, np.transpose(U))


def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32))
    orig = np.transpose(orig)
    proj = np.reshape(proj, (32, 32))
    proj = np.transpose(proj)
    fig, (orig_display, proj_display) = plt.subplots(ncols=2, )
    orig_display.set_title("Original")
    orig_ret = orig_display.imshow(orig, aspect='equal')
    fig.colorbar(orig_ret, ax=orig_display)
    proj_display.set_title("Projection")
    proj_ret = proj_display.imshow(proj, aspect='equal')
    fig.colorbar(proj_ret, ax=proj_display)
    plt.show()


def main():
        x = load_and_center_dataset('YaleB_32x32.npy')
        S = get_covariance(x)
        Lambda, U = get_eig(S, 2)
        projection = project_image(x[0], U)
        print(projection)
        display_image(x[0], projection)


if __name__ == "__main__":
    main()
