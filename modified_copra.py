import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import pandas as pd
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
import community
def node_influence(G):
    node_influence = {}
    A = {}
    for node_i in G.nodes():
        A[node_i] = {}
        for node_j in G.nodes():
            if node_i == node_j:
                A[node_i][node_j] = 0
            elif G.has_edge(node_i, node_j):
                A[node_i][node_j] = 1
            else:
                A[node_i][node_j] = 0

    for node in G.nodes():
        di = G.degree(node)
        neighbors = list(G.neighbors(node))
        denominator = sum(
            [G.degree(n) ** 2 for n in neighbors]) + G.degree(node)
        numerator = di / denominator
        NI = numerator 
        node_influence[node] = NI
    sorted_nodes = sorted(node_influence, key=node_influence.get, reverse=True)

    average_influence = sum(node_influence.values()) / len(node_influence)
    dc = set()
    dc = {
        node for node in sorted_nodes if node_influence[node] > average_influence}
    return dc

def read_graph_from_file():
    # G = nx.karate_club_graph()
    facebook = pd.read_csv("data/facebook_combined.txt.gz",
                           compression="gzip",
                           sep=" ",
                           names=["start_node", "end_node"],)
    G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")
    sequence = node_influence(G)
    for node, label in G.nodes(data=True):
        if(node in sequence):
            label['old'] = {node: 1.0}
        else:
            label['old'] = {}
        label['new'] = {}
    return G

def copra_lpa(G, v):
    def propagate(node):
        new_label = defaultdict(float)
        new_label = G.nodes[node]['new']
        nb_label_count = {}
        nb_degree_label_count = {}
        for neighbor in G.neighbors(node):
            old_label = G.nodes[neighbor]['old']
            for label in old_label:
              new_label[label] = new_label.get(label, 0.0) + old_label[label]
              nb_label_count[label] = nb_label_count.get(label, 0.0)+1
              nb_degree_label_count[label] = nb_degree_label_count.get(
                    label, 0.0)+G.degree(neighbor)
        normalize(new_label)
        threshold = 1.0 / v
        del_labels, coefficient_max = set(), 0.0
        li_score = {}
        for label, coefficient in list(new_label.items()):
            neighbors = set(G.neighbors(node))
            numerator = nb_label_count[label]
            denominator = len(neighbors)            
            li_score[label] = numerator / denominator 

            if coefficient < threshold:
                del new_label[label]
                if coefficient > coefficient_max:
                    del_labels.clear()
                    del_labels.add(label)
                    coefficient_max = coefficient
                elif coefficient == coefficient_max:
                    del_labels.add(label)

        if len(new_label) == 0 and len(del_labels) == 0:
            return
        max_label_score = 0
        max_label = 0
        for label in del_labels:
            if li_score[label] > max_label_score:
                max_label = label
        if len(new_label) == 0:
            new_label[max_label] = 1.0
        else:
            normalize(new_label)
            
    def normalize(labels):
        sum_val = sum(labels.values())
        if sum_val == 1:
            return
        for label in labels:
            labels[label] = labels[label] / sum_val

    def reset_current_label():
        for node in G.nodes():
            G.nodes[node]['old'] = G.nodes[node]['new']
            G.nodes[node]['new'] = {}

    def label_set(label_type):
        labels = set()
        for node in G.nodes():
            labels.update(G.nodes[node][label_type].keys())
        return labels

    def label_count(label_type):
        labels = {}
        for node in G.nodes():
            for label in G.nodes[node][label_type]:
                labels[label] = labels.get(label, 0) + 1
        return labels

    def mc(label_count_1, label_count_2):
        label_count = {}
        for label in label_count_1:
            label_count[label] = min(
                label_count_1[label], label_count_2[label])
        return label_count

    new_min = {}
    old_min = {}
    loop_count = 0

    while True:
        loop_count += 1
        print('loop', loop_count)
        for node in G.nodes():
            propagate(node)
        old = label_set('old')
        new = label_set('new')
        if old == new:
            new_min = mc(new_min, label_count('new'))
        else:
            new_min = label_count('new')
        if new_min != old_min:
            old_min = new_min
            reset_current_label()
            continue
        return

if __name__ == '__main__':
    g = read_graph_from_file()  
    copra_lpa(g, 3)
    true_labels = [float(list(labels.keys())[0]) for v, labels in g.nodes(data='new')]    
    node_color = []
    for v in g:
        labels = g.nodes[v]['new']
        if len(labels) == 1:
            node_color.append(float(list(labels.keys())[0]))
        else:
            node_color.append(5000)
    print(node_color)    
    labels = {node: node for node in g.nodes()}
    pos = nx.spring_layout(g, iterations=15, seed=1721)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
    nx.draw_networkx(g, node_color=node_color, pos=pos,
     ax=ax, **plot_options)
    plt.show()