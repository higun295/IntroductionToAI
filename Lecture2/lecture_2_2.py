from abc import ABC, abstractmethod
from collections import defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. 먼저 rollout 한 다음, 위치(move) 선택"
    def __init__(self, c=1):
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.c = c

    def choose(self, node):
        "node의 최선인 자식 노드 선택"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "게임 트리에서 한 층만 더 보기"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "node의 아직 시도해보지 않은 자식 노드 찾기"
        path = []
        