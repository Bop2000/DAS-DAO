import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, Set, Optional, Dict, List
from tqdm import tqdm

import numpy as np

from das_dao.objective_func import obj_function
from das_dao.pareto_front import pareto_frontier,pareto_evaluation

@dataclass
class TreeExploration:
    
    """
    Attributes:

        func (class): Objective function to evaluate the performance of new params.
        N (dict): Dict to record numebr of visit.
        children (dict): Dict to record leaf nodes information.
        rollout_round (int): Rollout times, i.e. expansion times of tree search.
        ratio (float): A larger ratio can lead to more exploration less exploitation.
        exploration_weight (float): Equals to ratio multiple max(score).
        num_list (list): Samples to be collected for a single tree rollout period,
            1st position number is top-scored samples, 
            2nd top-visited samples, 3rd random samples.
        n_tree (int): Num of single trees, 
            and will choose different root node for each tree.
        num_samples_per_acquisition (int): Num of final chosen samples in all trees.

    """
    
    func: Optional[obj_function] = None
    N: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    children: Dict[Any, Set] = field(default_factory=dict)
    rollout_round: int = 200
    ratio: float = 0.1
    exploration_weight: float = 0.1
    num_list: List[int] = field(default_factory=lambda: [5,3,1,1])
    n_tree: int = 10 # in which 80% trees select top scored sample as root node, 20% random
    num_samples_per_acquisition: int = 20


    def choose(self, node):
        """Choose the best successor of node."""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for trees"""
            return n.value + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n] + 1)
            )

        media_node = max(self.children[node], key=uct)
        node_rand = [
            list(self.children[node])[i].tup
            for i in np.random.randint(0, len(self.children[node]), 2)
        ]
        print(f"Best leaf node: {media_node}")
        return (
            (media_node, node_rand)
            if uct(media_node) > uct(node)
            else (node, node_rand)
        )

    def do_rollout(self, node):
        """Make the tree one layer better. (Train for one iteration.)"""
        self._expand(node)
        self._backpropagate(path=node)
    
    def _expand(self, node):
        """Update the `children` dict with the children of `node`"""
        self.children[node] = node.find_children(
            node, self.func
        )

    def _backpropagate(self, path):
        """Send the reward back up to the ancestors of the leaf"""
        self.N[path] += 1

    @staticmethod
    def data_process(x: np.ndarray, boards: List[list]) -> np.ndarray:
        new_x = []
        boards = np.unique(np.array(boards), axis=0)
        new_x = [board for board in boards if not np.any(np.all(board == x, axis=1))]
        print(f"Unique number of boards: {len(new_x)}")
        return np.array(new_x)

    def most_visit_node(self, x: np.ndarray, top_n: int) -> np.ndarray:
        """Find the most visited nodes."""
        N_visit = self.N
        childrens = [i for i in self.children]
        children_N = []
        X_top = []
        for child in childrens:
            child_tup = np.array(child.tup)
            same = np.all(child_tup == x, axis=1)
            has_true = any(same)
            if not has_true:
                children_N.append(N_visit[child])
                X_top.append(child_tup)
        children_N = np.array(children_N)
        X_top = np.array(X_top)
        ind = np.argpartition(children_N, -top_n)[-top_n:]
        X_topN = X_top[ind]
        return X_topN
    
    def top_score_node(self,
                       x: np.ndarray,
                       boards: np.ndarray,
                       top_n: int) -> np.ndarray:
        """Find the nodes with highest score."""
        new_x = self.data_process(x, boards)
        
        new_pred = self.func(new_x)
        new_pred = np.array(new_pred).reshape(len(new_x))
    
        ind = np.argsort(new_pred)[-top_n:]
        top_X = new_x[ind]
        return top_X
    
    def random_node(self,
                    x: np.ndarray,
                    boards_rand: np.ndarray,
                    n: int) -> np.ndarray:
        """Return random nodes."""
        new_rands = self.data_process(x, boards_rand)
        ind = np.random.choice(np.arange(len(new_rands)),n,replace=False)
        X_rand = new_rands[ind]
        return X_rand
    
    def pf_node(self,
                    x: np.ndarray,
                    boards_rand: np.ndarray,
                    n: int) -> np.ndarray:
        """Return random nodes."""
        new_rands = self.data_process(x, boards_rand)
        sample_score=self.func(np.array(new_rands).reshape(len(new_rands), -1, 1))
        whe=np.where(sample_score > max(sample_score) * 0.8)[0]
        new_rands2=new_rands[whe]
        sample_score2=sample_score[whe]
        print('length of new_rands2:',len(new_rands2))
        X_scaled_init=self.func.x_scaler.transform(x)
        X_scaled_sample=self.func.x_scaler.transform(new_rands2)
        ind2=pareto_evaluation(X_scaled_init, X_scaled_sample,sample_score2,n)
        top_pf = new_rands2[ind2]   
        return top_pf

    def single_rollout(self, X, board_uct, num_list: List[float]):
        """Perform a single rollout."""
        boards = []
        boards_rand = []
        for i in tqdm(range(self.rollout_round)):
            print("")
            print("="*10)
            print("Rollout No.:", i+1)
            print("="*10)
            self.do_rollout(board_uct)
            board_uct, board_rand = self.choose(board_uct)
            boards.append(list(board_uct.tup))
            boards_rand.append(list(board_rand))

        # highest pred value nodes
        X_top_score = self.top_score_node(X, np.array(boards), num_list[0])
        
        # pareto front nodes of Euclidean distance versus Pred score in 'boards_rand'
        top_pf = self.pf_node(X, np.vstack(boards_rand), num_list[1])
        
        # most visit nodes
        X_most_visit = self.most_visit_node(X, num_list[2])

        # random nodes
        X_rand = self.random_node(X, np.vstack(boards_rand), num_list[3])

        return np.concatenate([X_top_score, top_pf, X_most_visit, X_rand])

    def rollout(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Perform rollout"""
        x_current_top = self._get_unique_top_points(x, y)
        x_top = []
        for initial_X in x_current_top:
            print('initial_X:',initial_X)
            self.exploration_weight = self.ratio * abs(max(y))
            board_uct = OptTask(
                tup=tuple(initial_X), 
                value=float(self.func(initial_X.reshape(1,-1))), 
                terminal=False
                )
            x_top.append(self.single_rollout(x, board_uct, self.num_list))
        # return np.vstack(x_top)[: self.num_samples_per_acquisition]
        return np.vstack(x_top)
    
   
    def _get_unique_top_points(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            ) -> np.ndarray:

        top_select = round(self.n_tree*0.8)
        random_select = round(self.n_tree*0.2)
        ind = np.argsort(y)[-top_select:]
        ind_random=np.setdiff1d(np.arange(len(y)), ind)
        ind2 = np.random.choice(ind_random,random_select)
        ind = np.concatenate((ind,ind2))
        
        x_current_top = X[ind]
        y_top=y[ind]
        print('y score of selected root nodes:',y_top)    

        return x_current_top

class Node(ABC):
    """
    A representation of a single board state.
    DOTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        """All possible successors of this board state"""
        return set()

    @abstractmethod
    def is_terminal(self):
        """Returns True if the node has no children"""
        return True

    @abstractmethod
    def __hash__(self):
        """Nodes must be hashable"""
        return 123456789

    @abstractmethod
    def __eq__(self, node2):
        """Nodes must be comparable"""
        return True


_OT = namedtuple("OptTask", "tup value terminal")


class OptTask(_OT, Node):
    """Represents an optimization task node in the search tree."""

    @staticmethod
    def find_children(board, func):
        """Find all possible child nodes for the current board state."""
        if board.terminal:
            return set()
        
        all_tuples = []
        for i,index in enumerate(func.design_params):
            tup = list(board.tup)
            flip = np.random.randint(0,6)
            
            if flip ==0:
                indices = np.argmin(abs(func.param_space[index]-tup[i]))
                try:
                    tup[i] = func.param_space[index][indices+1]
                except:
                    tup[i] = func.param_space[index][indices-1]

            elif flip ==1:
                indices = np.argmin(abs(func.param_space[index]-tup[i]))
                try:
                    tup[i] = func.param_space[index][indices-1]
                except:
                    tup[i] = func.param_space[index][indices+1]

            elif flip ==2:
                n_flip = np.random.choice(
                    np.arange(len(func.design_params)),
                    int(len(func.design_params)/3),
                    replace=False
                    )
                for n in n_flip:
                    tup[n] = np.random.choice(
                        func.param_space[func.design_params[n]],
                        1)[0]

            elif flip ==3:
                n_flip = np.random.choice(
                    np.arange(len(func.design_params)),
                    int(len(func.design_params)/5),
                    replace=False
                    )
                for n in n_flip:
                    tup[n] = np.random.choice(
                        func.param_space[func.design_params[n]],
                        1)[0]

            else:
                tup[i] = np.random.choice(func.param_space[index],1)[0]

            all_tuples.append([round(t, 8) for t in tup])

        all_values = func(
            x=np.array(all_tuples).reshape(len(all_tuples),func.dims,1),
            )
        print('Predicted score of leaf nodes:',all_values)

        return {OptTask(tuple(t), v, False) for t, v in zip(all_tuples, all_values)}

    def is_terminal(self):
        """Check if the current board state is terminal."""
        return self.terminal
    
