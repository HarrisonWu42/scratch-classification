import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from utils import load_json, write_json
import math


def embedding_teacher(root):
	info_path = root + 'info.csv'
	infos = pd.read_csv(info_path)
	similarity_path = root + 'feature/similarity.csv'
	similaritys = pd.read_csv(similarity_path)
	similaritys['id'] = similaritys['id'].astype('int')
	similaritys['label'] = similaritys['label'].astype('int')
	df = pd.merge(infos, similaritys, on='id', how='inner')
	df = df.drop(columns=['label_x'])
	df = df.rename(columns={'label_y': 'label'})
	df.to_csv(root + 'feature/similarity_teacher.csv', encoding='utf8', index=None)


def key_variable(root):
	filepath = root + 'feature/similarity_teacher.csv'
	data = pd.read_csv(filepath)

	scpath = root + 'source_code/*.json'
	filelist = glob.glob(scpath)
	feature = []

	for i in tqdm(range(len(filelist))):
		feat = {}
		filepath = filelist[i]
		id = filepath.split('\\')[-1].split('.')[0]  # 特征1：id(filename)
		feat['id'] = int(id)

		scode = load_json(filepath)
		targets = scode['targets']

		for e in targets:
			name = e['name']
			blocks = e['blocks']

			if name in "方块兽":
				for k, v in blocks.items():
					opcode = v['opcode']
					if opcode == "motion_pointtowards_menu":
						fields = v['fields']
						towards = fields['TOWARDS']
						towards_cond = towards[0]
						feat['towards_cond'] = towards_cond
					if opcode == "control_if":
						inputs = v['inputs']
						if 'CONDITION' in inputs.keys():
							condition_k = inputs['CONDITION']
							if condition_k[0] == 2:
								condition_bk = blocks[condition_k[1]]
								feat['fks_if_condition_opcode'] = condition_bk['opcode']
						if 'SUBSTACK' in inputs.keys():
							substack_k = inputs['SUBSTACK']
							if substack_k[0] != 1:
								substack_bk = blocks[substack_k[1]]
								feat['fks_if_substack_opcode'] = substack_bk['opcode']

		feature.append(feat)

	feat1 = pd.DataFrame(columns=('id', 'towards_cond', 'fks_if_condition_opcode', 'fks_if_substack_opcode'))
	for e in feature:
		feat1 = feat1.append([e], ignore_index=True)
	feat1 = feat1.sort_values(by='id', ascending=True)

	df = pd.merge(feat1, data, on='id')
	df.to_csv(root + 'feature/key_variable.csv', encoding='utf8', index=None)


def mccabe(root):
	filepath = root + 'feature/key_variable.csv'
	data = pd.read_csv(filepath)

	scpath = root + 'source_code/*.json'
	filelist = glob.glob(scpath)
	feature = []

	for i in tqdm(range(len(filelist))):
		feat = {}
		filepath = filelist[i]
		id = filepath.split('\\')[-1].split('.')[0]  # 特征1：id(filename)
		feat['id'] = int(id)

		score = 1  # McCabe度量法
		scode = load_json(filepath)
		targets = scode['targets']

		for e in targets:
			name = e['name']
			blocks = e['blocks']

			for k, v in blocks.items():
				try:
					opcode = v['opcode']
					if opcode == "operator_and":
						score += 1
					elif opcode == "operator_or":
						score += 1
					elif opcode == "control_repeat":
						score += 1
					elif opcode == "control_forever":
						score += 1
					elif opcode == "control_if":
						score += 1
					elif opcode == "control_if_else":
						score += 2
					elif opcode == "control_repeat_until":
						score += 1
				except:
					pass
		feat['mccabe_score'] = score

		feature.append(feat)

	feat1 = pd.DataFrame(columns=('id', 'mccabe_score'))
	for e in feature:
		feat1 = feat1.append([e], ignore_index=True)
	feat1 = feat1.sort_values(by='id', ascending=True)

	df = pd.merge(feat1, data, on='id')
	df.to_csv(root + 'feature/mccabe.csv', encoding='utf8', index=None)


def halstead(root):
	filepath = root + 'feature/mccabe.csv'
	data = pd.read_csv(filepath)

	scpath = root + 'source_code/*.json'
	filelist = glob.glob(scpath)
	feature = []

	for i in tqdm(range(len(filelist))):
		feat = {}
		filepath = filelist[i]
		id = filepath.split('\\')[-1].split('.')[0]  # 特征1：id(filename)
		feat['id'] = int(id)

		N1 = 0  # 唯一操作数总数
		N2 = 0  # 唯一操作符总数
		n1 = 0  # 操作数总数
		n2 = 0  # 操作符总数

		scode = load_json(filepath)
		targets = scode['targets']

		opcode_set = set()
		opnum_set = set()
		for e in targets:
			name = e['name']
			blocks = e['blocks']

			for k, v in blocks.items():
				try:
					opcode = v['opcode']
					n2 += 1
					opcode_set.add(opcode)
				except:
					pass
				try:
					inputs = v['inputs']
					n1 += len(inputs)
					for input in inputs:
						opnum_set.add(input)
				except:
					pass
		# print(n1, "", n2)
		N1 = len(opnum_set)
		N2 = len(opcode_set)
		N = N1 + N2
		n = n1 + n2
		if n != 0:
			V = N * np.log2(n)
			D = (n1 / 2) * (N2/n2)
			E = D * V
			T = E / 18.0
			T_correct = 1.42*T + 1250
			O = 0.007 * E + 46
		else:
			O = 0
			T = 0

		feat['op_amount'] = O
		feat['time_consuming'] = T

		feature.append(feat)

	feat1 = pd.DataFrame(columns=('id', 'op_amount', 'time_consuming'))
	for e in feature:
		feat1 = feat1.append([e], ignore_index=True)
	feat1 = feat1.sort_values(by='id', ascending=True)

	df = pd.merge(feat1, data, on='id')
	df.to_csv(root + 'feature/halstead.csv', encoding='utf8', index=None)


if __name__ == '__main__':
	# ubantu
	# root = "/home/wh/Project/Data/p4/"
	# win
	root = "D:/Workspace/Project/Data/p4/"
	# embedding_teacher(root)
	# key_variable(root)
	# mccabe(root)
	halstead(root)