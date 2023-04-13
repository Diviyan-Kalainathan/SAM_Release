import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
# Generate test_datasets
import matplotlib.pyplot as plt
from code.sam import gSAM3d

# np.random.seed(0)
plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('savefig', dpi=600)
plt.rc('axes', labelsize=16)
plt.rc("legend", fontsize=12)


def corr_caus(n):
    # True causal graph = A->B->C
    A = np.random.uniform(size=n)
    EB, EC = np.random.uniform(size=(2, n))
    B = .5 * A + EB
    C = B + EC

    data = pd.DataFrame()
    data["A"] = scale(A)
    data["B"] = scale(B)
    data["C"] = scale(C)
    return data


def v_structure(n):
    # True causal graph : A->C<-B, normal gaussian variables, gaussian noise.
    A, B, EC = np.random.normal(size=(3, n))
    C = A + B + .4*EC
    data = pd.DataFrame()
    data["A"] = A
    data["B"] = B
    data["C"] = C
    data = pd.DataFrame(scale(data.values), columns=data.columns)
    return data


def XOR(n):
    A = np.concatenate([np.random.normal(-1, .1, n/2),
                       np.random.normal(1, .1, n/2)], 0)
    B = np.concatenate([np.random.normal(-1, .1, n/2),
                       np.random.normal(1, .1, n/2)], 0)
    np.random.shuffle(B)
    C = A*B + np.random.uniform(-.2, .2, size=n)
    data = pd.DataFrame()
    data["A"] = A
    data["B"] = B
    data["C"] = C
    print(np.correlate(data["A"], data["C"]))
    print(np.correlate(data["B"], data["C"]))
    # print(data)
    return data


def linear_confounder(n):
    Z, E1, E2 = np.random.normal(size=(3, n))
    B = Z + 0.3*E1
    A = Z + 0.3*E2
    data = pd.DataFrame()
    data['A'] = A
    data['B'] = B
    return data


def hidden_confounder(n):
    # Donut
    X = np.random.uniform(0, 2 * np.pi, size=n)
    COSX = np.cos(X) + np.random.uniform(-.2, .2, size=n)
    SINX = np.sin(X) + np.random.uniform(-.2, .2, size=n)
    data = pd.DataFrame()
    data["cosx"] = COSX
    data["sinx"] = SINX
    return data


def parallelogram(n):
    A = np.random.uniform(size=n)
    B = A + np.random.uniform(-.3, .3, size=n)
    data = pd.DataFrame()
    data["A"] = A
    data["B"] = B
    return data


def linear_gaussian(n):
    A = np.random.normal(size=n)
    B = A + 0.3*np.random.normal(size=n)
    data = pd.DataFrame()
    data["A"] = A
    data["B"] = B
    return data


def parabola(n):
    A = np.random.uniform(0, 2, size=n)
    E = np.random.normal(0, 0.1, size=n)
    B = A**2 + E
    data = pd.DataFrame()
    data["A"] = scale(A)
    data["B"] = scale(B)
    return data

def doubleV(n):
    A = np.random.uniform(-1, 1, size=n)
    E = np.random.uniform(1, 1, size=n)
    B = 4*(A**2 - 0.5)**2 + E/3
    data = pd.DataFrame()
    data["A"] = scale(A)
    data["B"] = scale(B)
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # data = XOR()
    # plt.hist(data["A"], bins=30, label="A")
    # plt.hist(data["C"], bins=30, label="C")
    # plt.hist(data["B"], bins=30, label="B")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(3, 3))
    # plt.tick_params(
    #                 axis='both',          # changes apply to the x-axis
    #                 labelleft="off",
    #                 labelbottom='off')  # labels along the bottom edge are off
    # data = parabola(700)
    # plt.scatter(*data.as_matrix().transpose(), s=5, alpha=0.8)
    # plt.ylabel(r"B")
    # plt.tight_layout()
    # plt.show()
    
    parameters={'lr':[0.01],
               'dlr':[0.001],
               'lambda1': [0],
               'lambda2':[0.0001],
               'nh':[20],
               'numberHiddenLayersG':[2],
               'dnh': [200],
               'losstype':["fgan"],
               'functionalComplexity':["l2_norm"], #numberHiddenUnits, l2_norm
               'sampletype':["sigmoidproba"],
               'train_epochs':[3000],
               'test_epochs':[1000],
               'dagstart':[0.5],
               'dagloss':[True],
               'dagpenalization':[0],
               'dagpenalization_increase':[0.01],
               'use_filter':[False],
               'filter_threshold':[0.5],
               'linear':[False]}

    
    parameters = {k:parameters[k][0] for k in parameters}
    data = doubleV(500)
    print(data)

    data.to_csv("doubleV.csv")

    #model = gSAM3d(**parameters)

    #result = model.predict(data, nruns=16, njobs=16, gpus=1)
    #print(result)
