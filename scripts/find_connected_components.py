import networkx as nx

def connected_components(graph_list):
    """
    Encuentra los componentes conectados en cada grafo.

    Parameters:
    graph_list (list of networkx.Graph): Lista de grafos.

    Returns:
    tuple: Una tupla que contiene una lista de componentes conectados para cada grafo
           y la suma total de la cantidad de componentes conectados en todos los grafos.
    """
    
    print("\nFinding connected components...\n")

    cc_list = []  # Lista para almacenar los componentes conectados de cada grafo

    for i, graph in enumerate(graph_list):
        cc = []  # Lista para almacenar los componentes conectados del grafo actual
        
        # Obtenemos componentes conectados
        cc = list(nx.connected_components(graph))

        cc_list.append(cc)  # Añadir los componentes conectados a la lista

    # Calcular la suma total de la cantidad de componentes conectados en todos los grafos
    suma_total = sum(len(cc) for cc in cc_list)

    #print(f"Suma total de la cantidad de componentes conectados en cada grafo: {suma_total}")

    return cc_list, suma_total



