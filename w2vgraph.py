import scipy.io
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def load_word2vec_dataset():
                
    dataset = scipy.io.loadmat('dataset/wv_embeddings.mat')
    wv = np.squeeze(dataset['features'])  # word2vec features multi-label
    np.array(wv) 
    print("Training set (wv) shape: {shape}", wv.shape) 

    graph = np.zeros((125,125))
    for i in range(125):
       sum2 = 0 
       print(i)
       for j in range(125):
           for k in range(300):
               graph[i,j] =  sum2 + np.sqrt(np.square(wv[i,k] - wv[j,k]))

    print(graph)
    scipy.io.savemat('dataset/wv_graph.mat', {'graph':graph}) #saving
   
    return graph

#h = load_word2vec_dataset()

def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges)


def test_mst():
    P = np.random.uniform(size=(125, 2))

    X = scipy.io.loadmat('dataset/wv_graph.mat')
    X = np.squeeze(X['graph'])  # word2vec features multi-label
    edge_list = minimum_spanning_tree(X)
    plt.scatter(P[:, 0], P[:, 1])
    scipy.io.savemat('dataset/mst_graph.mat', {'edge':edge_list})
    
    for edge in edge_list:
        i, j = edge
        plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    plt.show()

if __name__ == "__main__":
    test_mst()
