import pandas as pd
import numpy as np
import networkx as nx
import glob
import time
from karateclub.graph_embedding import Graph2Vec
import torch


def read_data(path):
	graph_list = glob.glob(path + "*.gexf")
	graphs = []
	dic = {}
	for i in range(len(graph_list)):
		gpath = graph_list[i]
		gid = graph_list[i].split('/')[-1].split('.')[0]
		dic[gpath] = int(gid)
	dic_sorted = sorted(dic.items(), key=lambda item: item[1], reverse=True)

	graph_paths = []
	for item in dic_sorted:
		graph_paths.append(item[0])

	for i in range(len(graph_paths)):
		gpath = graph_paths[i]
		graph = nx.read_gexf(gpath, node_type=int)
		graphs.append(graph)
	return graphs


def graph2vec(root, dimensions, epochs):
	data_path = root + "graph/"
	info_path = root + 'info.csv'
	info = pd.read_csv(info_path)
	labels = info['label'].values.flatten()
	graphs = read_data(data_path)

	# # AST of scratch3 project with perfect score
	# graph_bases = []
	# for i in range(14):
	# 	graph_base_path = root + 'type1_base/base' + str(i+1) + '.gexf'
	# 	g = nx.read_gexf(graph_base_path, node_type=int)
	# 	graph_bases.append(g)
	# all_graphs = graph_bases + graphs

	model = Graph2Vec(attributed=True, dimensions=dimensions, epochs=epochs)
	model.fit(graphs)

	embedding = model.get_embedding()

	embedding_path = "./embedding/embedding_" + str(dimensions) + "_" + str(epochs) + ".pt"
	torch.save(embedding, embedding_path)
	return embedding_path


if __name__ == "__main__":
	root = "/home/wh/Project/Data/p4/"

	start = time.time()
	graph2vec(root, 8, 5)
	graph2vec(root, 16, 5)
	graph2vec(root, 32, 5)
	graph2vec(root, 64, 5)
	graph2vec(root, 128, 5)
	graph2vec(root, 256, 5)
	end = time.time()
	print("Running time：%.2f s" % (end - start))