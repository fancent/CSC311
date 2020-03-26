import scipy
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, run_knn

def main():
    train, valid, test, trainTarget, validTarget, testTarget = load_data("digits.npz")
    print(valid.shape)
    kRange = [2,5,10,20,30]

    m = np.mean(train , axis=0)
    train_centered = train - np.tile(m, (train.shape[0], 1))
    C = np.cov(train_centered.T)
    U,S,V = np.linalg.svd(C)
    plot_data = []
    for k in kRange:
        train_recon = train_centered.dot(U[:,:k])
        print(train_centered.shape, U[:,:k].shape)

        knn_prediction = run_knn(1, train_recon,trainTarget, np.dot(valid, U[:, :k]))
        
        acc = np.mean(knn_prediction == validTarget)
        print("validation accuracy with k={}: {}%".format(k,acc*100))
        plot_data.append(1-acc)

    # Test data performance
    knn_prediction = run_knn(1, train_recon,trainTarget, np.dot(test, U[:, :30]))
    acc = np.mean(knn_prediction == testTarget)
    print("test accuracy with k={}:".format(k), acc*100, "%")

    # Plotting
    plt.plot(kRange, np.array(plot_data)*100)
    plt.plot(kRange, np.array(plot_data)*100, 'x')
    plt.title("Classification error over different K values")
    plt.xlabel("K values")
    plt.ylabel("Classification error (%)")
    plt.grid()
    plt.savefig("q3.png")
    plt.show()

if __name__ == '__main__':
    main()
