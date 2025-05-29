import osmnx as ox

# Define the place you want to extract the network for
place_name = "London, UK"

# Download the street network for London
# network_type='drive' will include roads routable by car
G = ox.graph_from_place(place_name, network_type='drive')

# G is now a networkx.MultiDiGraph object
# You can access nodes and edges
nodes, edges = ox.graph_to_gdfs(G)

print(f"Number of nodes (junctions): {len(nodes)}")
print(f"Number of edges (roads): {len(edges)}")

# You can also visualize the graph
fig, ax = ox.plot_graph(G)

# Save the graph to a file
ox.save_graphml(G, filepath='london_street_network.graphml')

# Save the nodes and edges to shapefiles
#nodes.to_file('london_nodes.shp')
#edges.to_file('london_edges.shp')
