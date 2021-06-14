from utils import load_json, write_json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


def buildGraph(filepath):
    # project 第一层
    sb3 = load_json(filepath)
    targets = sb3['targets']
    monitors = sb3['monitors']
    extensions = sb3['extensions']
    meta = sb3['meta']

    G = nx.Graph(name="graph")  # 创建无向图
    # G = nx.DiGraph(name="graph")　# 创建有向图

    #     print("This project have", len(targets), "targets")
    stage_num = 0
    role_num = 0
    id = 0
    id2blockid = dict()
    blockid2id = dict()

    # 为所有Stage, role, block 构建映射
    for target in targets:
        isStage = target['isStage']
        if isStage is True:
            stage_num += 1
            name = target['name']
            if name not in blockid2id.keys():
                id2blockid[id] = name
                blockid2id[name] = id
                id += 1

            # Stage也有block
            blocks = target['blocks']
            for block_id in blocks:
                block = blocks[block_id]
                if block_id not in blockid2id.keys() and isinstance(block, dict):
                    id2blockid[id] = block_id
                    blockid2id[block_id] = id
                    id += 1
        else:
            role_num += 1
            name = target['name']
            if name not in blockid2id.keys():
                id2blockid[id] = name
                blockid2id[name] = id
                id += 1

            # Role's blocks
            blocks = target['blocks']
            for block_id in blocks:
                block = blocks[block_id]
                if block_id not in blockid2id.keys() and isinstance(block, dict):
                    id2blockid[id] = block_id
                    blockid2id[block_id] = id
                    id += 1

    #     print("Stage: ", stage_num)
    #     print("Role: ", role_num)
    #     print("Node: ", len(id2blockid))

    # 构建Graph
    for target in targets:
        isStage = target['isStage']
        if isStage == True:
            name = target['name']
            stage_id = blockid2id[name]
            G.add_node(stage_id, id=stage_id, feature=name, name=name, opcode="stage")  # 舞臺加入Graph

            blocks = target['blocks']
            for block_id in blocks:
                block = blocks[block_id]
                if isinstance(block, dict):
                    id = blockid2id[block_id]
                    opcode = block['opcode']
                    next = block['next']
                    parent = block['parent']
                    isToplevel = block['topLevel']
                    if isToplevel is True:
                        G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=stage_id)
                        G.add_edge(stage_id, id)
                    else:
                        if parent is None:
                            G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=stage_id)
                            G.add_edge(stage_id, id)
                        else:
                            parent_id = blockid2id[parent]
                            G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=parent_id)
                            G.add_edge(parent_id, id)
        else:
            name = target['name']
            role_id = blockid2id[name]
            G.add_node(role_id, id=role_id, name=name, feature=name, opcode="role", parent=stage_id)  # 將角色加入Graph
            G.add_edge(stage_id, role_id)

            blocks = target['blocks']
            for block_id in blocks:

                block = blocks[block_id]
                if isinstance(block, dict):
                    id = blockid2id[block_id]
                    opcode = block['opcode']
                    next = block['next']
                    parent = block['parent']
                    isToplevel = block['topLevel']
                    if isToplevel is True:
                        G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=role_id)
                        G.add_edge(role_id, id)
                    else:
                        if parent is None:
                            G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=role_id)
                            G.add_edge(role_id, id)
                        else:
                            parent_id = blockid2id[parent]
                            G.add_node(id, id=id, name=block_id, feature=opcode, opcode=opcode, parent=parent_id)
                            G.add_edge(parent_id, id)
    return G


if __name__ == "__main__":
    """
    为某json文件建图
    """
    # dir = "D:/Workspace/Project/Data/p4/"
    # dirpath = dir + "source_code/*.json"
    # graphpath = dir + "graph/"

    # dir = "/home/wh/Project/Data/p9/"
    # dirpath = dir + "source_code/*.json"
    # graphpath = dir + "graph/"
    # filelist = glob.glob(dirpath)
    #
    # for i in tqdm(range(len(filelist))):
    #     filepath = filelist[i]
    #     G = buildGraph(filepath)
    #     filename = filepath.split('/')[-1].split('.')[0]
    #     nx.write_gexf(G, graphpath + filename + ".gexf")


    G = buildGraph("C:/Users/hangzhouwh/Desktop/base14.json")
    nx.write_gexf(G, "C:/Users/hangzhouwh/Desktop/base14.gexf")