import networkx as nx

def find_components(graph_list):
    """
    Find the connected components in each graph.

    Parameters:
    graph_list (list of networkx.Graph): List of graphs.

    Returns:
    tuple: A tuple containing a list of connected components for each graph
           and the total sum of the number of connected components in all graphs.
    """
    
    cc_list = []

    for graph in graph_list:
        cc = list(nx.connected_components(graph))
        cc_list.append(cc)

    total_sum = sum(len(cc) for cc in cc_list)

    return cc_list, total_sum
