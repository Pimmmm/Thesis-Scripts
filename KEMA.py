import numpy as np
from scipy import linalg, sparse, stats
from tqdm.notebook import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity

import numpy.matlib

def kernel_manifold_alignment(data1, data2, mu, lanbda, n_neighbors, n_eigs, n_correspondence):
    """
    Aligns the manifolds of two datasets (d x n matrix), where d equals the number of dimensions and 
    n equals the amount of samples.
    
    Parameters:
    
    data1: column matrix (n_features x n_samples)
    data2: column matrix (n_features x n_samples)
    mu: float (0-1)
    lanbda: float (0-1)
    n_neighbors: the amount of neighbours to use when determining the topology Laplacian (integer)
    n_eigs: the number of dimensions of the latent feature space (integer)
    n_correpspondence: the number of datapoints having a correspondence in both datasets (integer)
    
    Returns:
    
    Phi1TtoF: column matrix of data1 transformed to the latent feature space (n_eigs, n_samples)
    Phi2TtoF: column matrix of data2 transformed to the latent feature space (n_eigs, n_samples)
    ALPHA: eigenvectors
    LAMBDA: eigenvalues
    """
    
    d1, n1 = data1.shape
    d2, n2 = data2.shape
    
    tot_samples = n1 + n2
    
    # build topography Laplacian
    print('Computing neighbours')
    x1_graph = np.zeros((n1, n1))
    for n in trange(n1):
        x1_nn = cosine_similarity(data1.T, data1.T[n].reshape(1,-1))
        x1_nn_ixs = x1_nn.argsort(axis=0)[::-1][1:n_neighbors+1].flatten()
        x1_graph[n,x1_nn_ixs] = 1
        
    x2_graph = np.zeros((n2, n2))
    for c in trange(n2):
        x2_nn = cosine_similarity(data2.T, data2.T[c].reshape(1,-1))
        x2_nn_ixs = x2_nn.argsort(axis=0)[::-1][1:n_neighbors+1].flatten()
        x2_graph[c, x2_nn_ixs] = 1
    
    print("Building graph Laplacians")
    #topology matrix
    W = linalg.block_diag(x1_graph, x2_graph)
    W = (W + W.T)/2
    
    # similarity matrix
    Ws = np.zeros((tot_samples, tot_samples))

    Ws1 = np.eye(n1)
    Ws2 = np.eye(n2)
    Ws3 = np.eye(n_correspondence)

    Ws[:n1, :n1] = Ws1
    Ws[n1:, n1:] = Ws2
    Ws[n1:(n1+n_correspondence), :n_correspondence] = Ws3
    Ws[:n_correspondence, n1:(n1+n_correspondence)] = Ws3

    Ws = Ws + np.eye(tot_samples)
    
    #dissimilarity matrix
    Wd = np.ones((tot_samples, tot_samples))
    np.fill_diagonal(Wd, 0)

    Wd1 = np.ones((n_correspondence, n_correspondence))
    np.fill_diagonal(Wd1, 0)

    Wd[:n_correspondence, n1:(n1+n_correspondence)] = Wd1
    Wd[n1:(n1+n_correspondence), :n_correspondence] = Wd1

    Wd = Wd + np.eye(Wd.shape[0])
    
    #normalize the (dis)similarity matrices
    Sws = sum(sum(Ws))
    Swd = sum(sum(Wd))
    Sw = sum(sum(W))

    Ws = Ws / Sws * Sw
    Wd = Wd / Swd * Sw
    
    #extract diagonals from the matrices
    Dd = np.diag(np.sum(Wd, axis=1))
    Ds = np.diag(np.sum(Ws, axis = 1))
    D = np.diag(np.sum(W, axis=1))
    
    #build Laplacians
    Ls = Ds - Ws # graph Laplacian of similarity 
    Ld = Dd - Wd # Laplacian of dissimilarity
    L = D - W # Laplacian topology/geometry
    
    #tweak the algorithm
    A = ((1 - mu) * L + mu * Ls) + lanbda * np.eye(Ls.shape[0])
    B = Ld
    
    #compute kernels
    kernel_data1 = np.matmul(data1.T, data1)
    kernel_data2 = np.matmul(data2.T, data2)
    
    K = linalg.block_diag(kernel_data1, kernel_data2)
    
    KA = np.matmul(K,A)
    KB = np.matmul(K,B)
    KAK = np.matmul(KA, K)
    KBK = np.matmul(KB, K)
    
    print("Solving generalized eigenvalue decomposition")
    #determine matrix rank
    rank_A = np.linalg.matrix_rank(KAK)
    rank_B = np.linalg.matrix_rank(KBK)
    
    ALPHA, LAMBDA,n_eig = gen_eig(KAK, KBK, 'LM', n_eigs, rank_A, rank_B)
    
    print("Rotating axis if needed")
    lambda_idxs = np.diag(LAMBDA).argsort()
    LAMBDA = np.sort(np.diag(LAMBDA))
    LAMBDA = LAMBDA.reshape(LAMBDA.shape[0],1)
    
    ALPHA = ALPHA[:, lambda_idxs]
    
    E1 = ALPHA[:n1, :] #eigenvectors for the first dataset (CAV)
    E2 = ALPHA[n1:, :] #eigenvectors for the second dataset (GloVe)
    
    #Compare the rotated axis with the 'normal' axis
    sourceXpInv = (-1 * np.matmul(E1.T, kernel_data1)).T
    sourceXp = np.matmul(E1.T, kernel_data1).T
    targetXp = np.matmul(E2.T, kernel_data2).T
    
    sourceXpInv = stats.zscore(sourceXpInv)
    sourceXp = stats.zscore(sourceXp)
    targetXp = stats.zscore(targetXp)
    
    ErrRec = np.zeros((n1, ALPHA.shape[1]))
    ErrRecInv = np.zeros((n1, ALPHA.shape[1]))
    
    m1 = np.zeros((n1, ALPHA.shape[1]))
    m1inv = np.zeros((n1, ALPHA.shape[1]))
    m2 = np.zeros((n1, ALPHA.shape[1]))
    
    for j in range(ALPHA.shape[1]):
        for i in range(n1):
            m1inv[i,j] = np.mean(sourceXpInv[i, j])
            m1[i,j] = np.mean(sourceXp[i, j])
            m2[i,j] = np.mean(targetXp[i, j])

            ErrRec[i,j] = np.square(np.power(np.mean(sourceXp[i, j]) - np.mean(targetXp[i, j]), 2))

            ErrRecInv[i,j] = np.square(np.power(np.mean(sourceXpInv[i, j]) - np.mean(targetXp[i, j]),2))
            
    Sc = ErrRec.max(axis=0) > ErrRecInv.max(axis=0)
    
    print("Inverting axis")
    
    ALPHA[:n1, Sc] = ALPHA[:n1, Sc] * -1
    
    Nf = 100
    nVectLin = min(Nf, rank_B)
    nVectLin = min(nVectLin, rank_A)
    
    T1 = n1
    T2 = n1
    
    print("Transforming data to the common feature space")
    for Nf in range(nVectLin):
        E1 = ALPHA[:n1, :Nf+1]
        E2 = ALPHA[n1:, :Nf+1]

        Phi1toF = np.matmul(E1.T, kernel_data1)
        Phi2toF = np.matmul(E2.T, kernel_data2)

        Phi1TtoF = np.matmul(E1.T, kernel_data1) 
        Phi2TtoF = np.matmul(E2.T, kernel_data2) 

        m1 = np.mean(Phi1toF.T, axis = 0)
        m2 = np.mean(Phi2toF.T, axis = 0)
        s1 = np.std(Phi1toF.T, axis = 0)
        s2 = np.std(Phi2toF.T, axis =0)

        Phi1TtoF = np.divide((Phi1TtoF.T - np.matlib.repmat(m1, T1, 1)), 
                             np.matlib.repmat(s1, T1, 1)).T

        Phi2TtoF = np.divide((Phi2TtoF.T - np.matlib.repmat(m2, T2 ,1)),
                             np.matlib.repmat(s2, T2, 1)).T
        
    return Phi1TtoF, Phi2TtoF, ALPHA, LAMBDA
    
def gen_eig(A, B, option, n_eig, rankA, rankB):
    """
    Extracts generalized eigenvalues for problem A * U = B * U * landa
    """
    
    
    n_eig = min(n_eig, rankA, rankB)
    
    B = (B + B.T) / 2
    R = B.shape[0]
    rango = rankB
    
    if rango == R:
        U = np.zeros((R, n_eig))
        D = np.zeros((n_eig, n_eig))
        inv_B = np.linalg.inv(B)
        for k in tqdm(range(n_eig)):
            d, a = sparse.linalg.eigs(np.matmul(inv_B, A),1, which=option) #'a' are the eigenvectors in the matlab code
            d = d.real
            a = a.real
            
            ab = np.matmul(a.T, B)
            a = np.divide(a, np.sqrt(np.matmul(ab, a)))
            U[:,k] = a.flatten()
            D[k,k] = d
            
            ba = np.matmul(B, a)
            aTb = np.matmul(a.T, B)
            dba = d * ba
            A = A - np.matmul(dba, aTb)
        
        return U, D, n_eig
    
    else:
        print('Calculating d and v')
        d, v = sparse.linalg.eigs(B, rango)
        d = d.real
        v = v.real
        
        B = np.matmul(v.T, B)
        B = np.matmul(B, v)
        
        A = np.matmul(v.T, A)
        A = np.matmul(A, v)
        
        U2 = np.zeros((rango, n_eig))
        D = np.zeros((n_eig, n_eig))
        print('Calculation inverse of B')
        inv_B = np.linalg.inv(B)
        
        for k in tqdm(range(n_eig)):
            
            d, a = sparse.linalg.eigs(np.matmul(inv_B, A),1, which=option)
            d = d.real
            a = a.real
            
            ab = np.matmul(a.T, B)
            aba = np.matmul(ab, a)
            a = np.divide(a, np.sqrt(aba))
            
            U2[:,k] = a.flatten()
        
            D[k,k] = d
            
            ba = np.matmul(B, a)
            aTb = np.matmul(a.T, B)
            
            dba = d * ba
            A = A - np.matmul(dba, aTb)
        
        U = np.matmul(v, U2)
        return U, D, n_eig

import numpy as np
import numpy.matlib
from scipy import linalg, sparse, stats
from tqdm.notebook import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity

def manifold_alignment_wang(data1, data2, mu, lanbda, n_neighbors, n_eigs, n_correspondence, n_cavs):

    d1, n1 = data1.shape
    d2, n2 = data2.shape

    T1 = n1
    T2 = n2
    
    tot_samples = n1 + n2
    print("Shape of data1:", data1.shape)
    print("Shape of data2:", data2.shape)
    
    # Transform data to normal distribution
    mean_data1 = np.mean(data1.T, axis = 0)
    mean_data2 = np.mean(data2.T, axis = 0)
    std_data1 = np.std(data1.T, axis = 0)
    std_data2 = np.std(data2.T, axis =0)


    data1 = np.divide((data1.T - np.matlib.repmat(mean_data1, T1, 1)), 
                         np.matlib.repmat(std_data1, T1, 1)).T

    data2 = np.divide((data2.T - np.matlib.repmat(mean_data2, T2 ,1)),
                         np.matlib.repmat(std_data2, T2, 1)).T
    
    Z = linalg.block_diag(data1, data2) #dimensions (d1+d2, n1+n2)
    
    # create k-Nearest neighbor graph for data structures 
    print("Computing neighbors dataset 1")

    x1_graph = np.zeros((n1, n1))

    #get the nearest neighbors of the data to the CAVS
    for n in trange(n1):
        x1_cavs_nn = cosine_similarity(data1.T[:n_cavs], data1.T[n].reshape(1,-1))

        if n < n_cavs:
            x1_cavs_nn_ixs = x1_cavs_nn.argsort(axis=0)[::-1][1:n_neighbors+1].flatten()
        else:
            x1_cavs_nn_ixs = x1_cavs_nn.argsort(axis=0)[::-1][:n_neighbors+10].flatten()

        x1_graph[n,x1_cavs_nn_ixs] = 1

    # force CAVs to indicate their nearest image neighbours
    for t in trange(n1):
        x1_imgs_nn = cosine_similarity(data1.T[n_cavs:], data1.T[t].reshape(1,-1))
        if t < n_cavs:
            x1_imgs_nn_ixs = x1_imgs_nn.argsort(axis=0)[::-1][:n_neighbors].flatten()

        else:
            x1_imgs_nn_ixs = x1_imgs_nn.argsort(axis=0)[::-1][1:n_neighbors+1].flatten()

        x1_imgs_nn_ixs += n_cavs
        x1_graph[t, x1_imgs_nn_ixs] = 1

    print('Computing neighbors dataset 2')
    x2_graph = np.zeros((n2, n2))
    for c in trange(n2):
        x2_nn = cosine_similarity(data2.T, data2.T[c].reshape(1,-1))
        x2_nn_ixs = x2_nn.argsort(axis=0)[::-1][1:n_neighbors+1].flatten()
        x2_graph[c, x2_nn_ixs] = 1
        if not np.array_equal(np.where(x2_graph[c] == 1)[0], np.sort(x2_nn_ixs)):
            print('The neighbors of dataset 2 do not correspond')
    
    
    # Computing the Laplacians
    print('Building Laplacians')
    # create topology Laplacian
    W = linalg.block_diag(x1_graph, x2_graph)
    W = (W + W.T)/2

    # create simmilarity matrix
    Ws = np.zeros((tot_samples, tot_samples))

    Ws1 = np.eye(n1)
    Ws2 = np.eye(n2)
    Ws3 = np.eye(n_correspondence)

    Ws[:n1, :n1] = Ws1
    Ws[n1:, n1:] = Ws2
    Ws[n1:(n1+n_correspondence), :n_correspondence] = Ws3
    Ws[:n_correspondence, n1:(n1+n_correspondence)] = Ws3

    Ws = Ws + np.eye(Ws.shape[0])

    #create dissimilarity matrix
    Wd = np.ones((tot_samples, tot_samples))
    np.fill_diagonal(Wd, 0)

    Wd1 = np.ones((n_correspondence, n_correspondence))
    np.fill_diagonal(Wd1, 0)

    Wd[:n_correspondence, n1:(n1+n_correspondence)] = Wd1
    Wd[n1:(n1+n_correspondence), :n_correspondence] = Wd1

    Wd = Wd + np.eye(Wd.shape[0])

    # normalize data
    Sws = sum(sum(Ws))
    Swd = sum(sum(Wd))
    Sw = sum(sum(W))

    Ws = Ws / Sws * Sw
    Wd = Wd / Swd * Sw

    # extract the diagonals
    Dd = np.diag(np.sum(Wd, axis=1))
    Ds = np.diag(np.sum(Ws, axis = 1))
    D = np.diag(np.sum(W, axis=1))

    # create laplacians
    Ls = Ds - Ws # graph Laplacian of similarity 
    Ld = Dd - Wd # Laplacian of dissimilarity
    L = D - W # Laplacian topology/geometry
    
    # tune the generalized eigenproblem
    A = ((1 - mu) * L + mu * Ls) + lanbda * np.eye((Ls.shape[0]))
    B = Ld

    ZA = np.matmul(Z, A)
    ZB = np.matmul(Z, B)

    ZAZ = np.matmul(ZA, Z.T)
    ZBZ = np.matmul(ZB, Z.T)
    
    rank_A = np.linalg.matrix_rank(ZAZ)
    rank_B = np.linalg.matrix_rank(ZBZ)
    
    print("Solving generalized eigenvalue decomposition")
    # V = eigenvectors, D = eigenvalues
    V, D, n_eig = gen_eig(ZAZ, ZBZ, 'LM', n_eigs, rank_A, rank_B)
    
    D_idxs = np.diag(D).argsort()
    D = np.sort(np.diag(D))
    D = D.reshape(D.shape[0],1)
    V = V[:, D_idxs]

    print("Rotating axis if needed")
    #rotate axis if needed
    E1 = V[:d1,:]
    E2 = V[d1:,:]

    sourceXpInv = (-1 * np.matmul(E1.T, data1)).T
    sourceXp = np.matmul(E1.T, data1).T
    targetXp = np.matmul(E2.T, data2).T

    sourceXpInv = stats.zscore(sourceXpInv)
    sourceXp = stats.zscore(sourceXp)
    targetXp = stats.zscore(targetXp)

#     ErrRec = np.zeros((n_correspondence, V.shape[1]))
#     ErrRecInv = np.zeros((n_correspondence, V.shape[1]))

#     m1 = np.zeros((n_correspondence, V.shape[1]))
#     m1inv = np.zeros((n_correspondence, V.shape[1]))
#     m2 = np.zeros((n_correspondence, V.shape[1]))

#     cls = np.arange(n_correspondence)
    
#     for j in trange(V.shape[1]):
#         for i in range(n_correspondence):
#             m1inv[i,j] = np.mean(sourceXpInv[cls[i], j])
#             #print('m1inv: ', m1inv)
#             m1[i,j] = np.mean(sourceXp[cls[i], j])
#             #print('m1: ', m1)
#             m2[i,j] = np.mean(targetXp[cls[i], j])
#             #print('m2: ', m2)

#             ErrRec[i,j] = np.square(np.power(np.mean(sourceXp[cls[i], j]) - np.mean(targetXp[cls[i], j]), 2))

#             ErrRecInv[i,j] = np.square(np.power(np.mean(sourceXpInv[cls[i], j]) - np.mean(targetXp[cls[i], j]),2))
    
    #store cosine similarity values 
    
    cs1 = np.zeros((n_correspondence, V.shape[1]))
    cs1_inv = np.zeros((n_correspondence, V.shape[1]))
        
    for j in trange(V.shape[1]):
        E1_inv = np.copy(E1)
        E1_inv[:,j] = -1*E1_inv[:,j]
        
        sourceInv = np.matmul(E1_inv.T, data1).T
        sourceInv = stats.zscore(sourceInv)
        
        for i in range(n_correspondence):
            cs1[i,j] = cosine_similarity(sourceXp[i].reshape(1,-1), targetXp[i].reshape(1,-1))
            cs1_inv[i,j] = cosine_similarity(sourceInv[i].reshape(1,-1), targetXp[i].reshape(1,-1))
        
    ErrRec = np.mean(cs1, axis = 0)
    ErrRecInv = np.mean(cs1_inv, axis=0)
    
    Sc = ErrRec < ErrRecInv
    V[:d1, Sc] = V[:d1, Sc] * -1
    
    Nf = d1+d2
    E1 = V[:d1, :Nf]
    E2 = V[d1:, :Nf]

    X1toF = np.matmul(E1.T, data1)
    X2toF = np.matmul(E2.T, data2)

    m1 = np.mean(X1toF.T, axis = 0)
    m2 = np.mean(X2toF.T, axis = 0)
    s1 = np.std(X1toF.T, axis = 0)
    s2 = np.std(X2toF.T, axis =0)


    XT1toF = np.divide((X1toF.T - np.matlib.repmat(m1, T1, 1)), 
                         np.matlib.repmat(s1, T1, 1)).T

    XT2toF = np.divide((X2toF.T - np.matlib.repmat(m2, T2 ,1)),
                         np.matlib.repmat(s2, T2, 1)).T
    
    return XT1toF, XT2toF, V, D, m1, s1

def FindNeighbours(A, n_neighbours, B=None, include_self=False):
    """
    A: matrix (n_features, n_samples_A)
    n_neighbours: number of neighbours (int)
    B: matrix (n_features, n_samples_B)
    
    Returns:
    neighbor_matrix (n_samples_A x n_samples_A) if B=None, otherwise (n_samples_B x n_samples_A)
    """
    
    if type(B) != numpy.ndarray:
        cosine_dist = cosine_similarity(A.T, A.T)
    else:
        cosine_dist = cosine_similarity(A.T, B.T)
    
    neighbor_matrix = np.zeros(cosine_dist.shape)
    
    for i in range(cosine_dist.shape[1]):
        cos_idx = cosine_dist[:,i].argsort(axis=0)[::-1].flatten()
        if include_self:
            top_nn = cos_idx[:n_neighbours]
        else:
            top_nn = cos_idx[1:n_neighbours+1]
        neighbor_matrix[top_nn, i] = 1
    
    return neighbor_matrix
    