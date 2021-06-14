"""
This file is used to generate an image from the AST parsed by the Scratch3 project.
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import colorsys
import random
import glob
from utils import load_json, write_json


def get_n_hls_colors(num):
	hls_colors = []
	i = 0
	step = 360.0 / num
	while i < 360:
		h = i
		s = 90 + random.random() * 10
		l = 50 + random.random() * 10
		_hlsc = [h / 360.0, l / 100.0, s / 100.0]
		hls_colors.append(_hlsc)
		i += step

	return hls_colors


def ncolors(num):
	rgb_colors = []
	if num < 1:
		return rgb_colors
	hls_colors = get_n_hls_colors(num)
	for hlsc in hls_colors:
		_r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
		r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
		rgb_colors.append([r, g, b])

	return rgb_colors


def color(value):
	digit = list(map(str, range(10))) + list("ABCDEF")
	if isinstance(value, tuple):
		string = '#'
		for i in value:
			a1 = i // 16
			a2 = i % 16
			string += digit[a1] + digit[a2]
		return string
	elif isinstance(value, str):
		a1 = digit.index(value[1]) * 16 + digit.index(value[2])
		a2 = digit.index(value[3]) * 16 + digit.index(value[4])
		a3 = digit.index(value[5]) * 16 + digit.index(value[6])
		return (a1, a2, a3)


def buildGraph(filepath):
	# project layer 1
	sb3 = load_json(filepath)
	targets = sb3['targets']
	G = nx.Graph(name="graph")  # create an undirected graph
	id = 0
	id2blockid = dict()
	blockid2id = dict()

	# create mapping for all stages, roles, and blocks
	for target in targets:
		isStage = target['isStage']
		if isStage is True:
			name = target['name']
			if name not in blockid2id.keys():
				id2blockid[id] = name
				blockid2id[name] = id
				id += 1

			# stage's block
			blocks = target['blocks']
			for block_id in blocks:
				block = blocks[block_id]
				if block_id not in blockid2id.keys() and isinstance(block, dict):
					id2blockid[id] = block_id
					blockid2id[block_id] = id
					id += 1
		else:
			name = target['name']
			if name not in blockid2id.keys():
				id2blockid[id] = name
				blockid2id[name] = id
				id += 1

			# role's blocks
			blocks = target['blocks']
			for block_id in blocks:
				block = blocks[block_id]
				if block_id not in blockid2id.keys() and isinstance(block, dict):
					id2blockid[id] = block_id
					blockid2id[block_id] = id
					id += 1

	# create a graph
	for target in targets:
		isStage = target['isStage']
		if isStage is True:
			name = target['name']
			stage_id = blockid2id[name]
			G.add_node(stage_id, feature=checkFeature["stage"], opcode="stage")

			blocks = target['blocks']
			for block_id in blocks:
				block = blocks[block_id]
				if isinstance(block, dict):
					id = blockid2id[block_id]
					opcode = block['opcode']
					parent = block['parent']
					isToplevel = block['topLevel']
					if isToplevel is True:
						G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
						G.add_edge(stage_id, id)
					else:
						if parent is None:
							G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
							G.add_edge(stage_id, id)
						else:
							parent_id = blockid2id[parent]
							G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
							G.add_edge(parent_id, id)
		else:
			name = target['name']
			role_id = blockid2id[name]
			G.add_node(role_id, id=role_id, feature=checkFeature["role"], opcode="role")  # add role to Graph
			G.add_edge(stage_id, role_id)

			blocks = target['blocks']
			for block_id in blocks:
				block = blocks[block_id]
				if isinstance(block, dict):
					id = blockid2id[block_id]
					opcode = block['opcode']
					parent = block['parent']
					isToplevel = block['topLevel']
					if isToplevel is True:
						G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
						G.add_edge(role_id, id)
					else:
						if parent is None:
							G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
							G.add_edge(role_id, id)
						else:
							parent_id = blockid2id[parent]
							G.add_node(id, feature=checkFeature[opcode], opcode=opcode)
							G.add_edge(parent_id, id)

	G.nodes[0]['depth'] = 0
	depth_dic = {}
	depth_dic[0] = 0
	bfs_path = list(nx.bfs_edges(G, source=0))
	for e in bfs_path:
		node1 = e[0]
		node2 = e[1]
		node1_depth = depth_dic[node1]
		node2_depth = node1_depth + 1
		depth_dic[node2] = node2_depth
		G.nodes[node2]['depth'] = node2_depth
	return G


# mapping corresponding to each code block
checkFeature = {}

dirpath = "../data/source_code/*.json"  # the directory path of the source code file of the sb3 file
filelist = glob.glob(dirpath)

opcode_set = set()
for i in tqdm(range(len(filelist))):
	filepath = filelist[i]
	scode = load_json(filepath)
	targets = scode['targets']
	for e in targets:
		name = e['name']
		blocks = e['blocks']
		for k, v in blocks.items():
			try:
				opcode = v['opcode']
				opcode_set.add(opcode)
			except:
				pass

opcode_set = list(opcode_set)
opcode_set.append("stage")
opcode_set.append("role")
i = 0
for op in opcode_set:
	checkFeature[op] = i
	i += 1

colors = list(map(lambda x: color(tuple(x)), ncolors(len(opcode_set))))
# print(len(opcode_set))
# print(len(colors))

graphs = []
for i in tqdm(range(len(filelist))):
	filepath = filelist[i]
	g = buildGraph(filepath)
	graphs.append(g)

# label
labelpath = '../data/info.csv'
labels = pd.read_csv(labelpath)['label'].values.tolist()

index = 0
for graph in tqdm(graphs):
	DG = graph
	root = list(DG.nodes())[0]
	dfs_seqs = list(nx.dfs_tree(DG, root))

	array = np.ndarray((250, 250, 3), np.uint8)
	array[:, :, 0] = 255
	array[:, :, 1] = 255
	array[:, :, 2] = 255
	image = Image.fromarray(array)
	draw = ImageDraw.Draw(image)

	num = 0
	for node in dfs_seqs:
		draw.rectangle((DG.nodes()[node]['depth'] * 10, num * 10, DG.nodes()[node]['depth'] * 10 + 70, num * 10 + 10),
					   colors[checkFeature[DG.nodes()[node]['opcode']]], 'black')
		num += 1

	image.save('./images/' + str(index) + '_' + str(labels[index]) + '.jpg')
	index += 1
