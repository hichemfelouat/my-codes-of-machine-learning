import networkx as nx
import random
G = nx.gnm_random_graph(10,25)
# random weighted 
for (u, v) in G.edges():
  G.edges[u,v]['weight'] = random.randint(0,99)/100
pos=nx.spring_layout(G)
labels = nx.get_edge_attributes(G,'weight')
nx.draw(G,pos)
nx.draw_networkx_edge_labels(G,pos,node_size=300, with_labels=True,edge_labels=labels)
nx.draw(G)
print("list nodes : ",G.nodes(),"list edges : ",G.edges(data=True))
# Betweenness centrality
Bc_nodes = nx.betweenness_centrality(G,normalized=True, weight= "weight")
clu_nodes = nx.clustering(G,weight= "weight")
squ_nodes = nx.square_clustering(G)
# Closeness centrality
clo_cen = nx.closeness_centrality(G)
# Eigenvector centrality
eig_cen = nx.eigenvector_centrality(G)
# Degree centrality
deg_cen = nx.degree_centrality(G)
# Betweenness centrality edges
bc_edges = nx.edge_betweenness_centrality(G,k=6,normalized=True, weight= "weight")
print("Nodes : Betweenness= ",Bc_nodes," clustering= ",clu_nodes," S-clustering= ",squ_nodes,
" closeness= ",clo_cen," Eigenvector= ",eig_cen," Degree= ",deg_cen," Edges : Betweenness= ",bc_edges)
