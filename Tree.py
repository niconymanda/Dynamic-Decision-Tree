from typing import List
import numpy as np

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 c: int = 0,
                 h: int = 1,
                 min_split_points: int = 1,
                 beta: float = 0,
                 current_h: int = 0):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        # Optimisation: minimisation of memory usage by storing only necessary information
        self.features = features
        self.labels = labels
        self.types = types
        self.h = h
        self.min_split_points = min_split_points
        self.current_h = current_h
        self.beta = beta
        self.c = c

        if current_h < h and len(set(self.labels)) > 1 and len(self.labels) >= min_split_points:
        
            features_left, features_right, labels_left, labels_right, self.unique_left, self.unique_right, self.feature_id = PointSet.split_node(self)
            
            self.left = Tree(features = features_left, labels = labels_left, types = types, h = h, min_split_points = min_split_points, current_h = current_h + 1)
            self.right = Tree(features = features_right, labels = labels_right, types = types, h = h, min_split_points = min_split_points, current_h = current_h + 1)

        else:
            self.left = None
            self.right = None

        # raise NotImplementedError('Implement this method for Question 4')

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                guessed by the Tree
        """
        # Optimisation: check if none before computing unnecessary computations. 
        if self.left is None and self.right is None:
            return sum(self.labels)/len(self.labels) >= 1/2
        else:
            
            if self.types[self.feature_id] != FeaturesTypes.REAL:
                if features[self.feature_id] == self.unique_left and self.left:
                    return self.left.decide(features)
                
                if features[self.feature_id] != self.unique_left and self.right:
                    return self.right.decide(features)
                
            else:
                if features[self.feature_id] <= self.unique_left and self.left:
                    return self.left.decide(features)
                
                if features[self.feature_id] > self.unique_left and self.right:
                    return self.right.decide(features)
                
        # raise NotImplementedError('Implement this method for Question 4')
    

    def aux_lookup(self, features, path):
        if self.left is None and self.right is None:
            return path
        else:
            current_feature_id = path[-1].feature_id if path[-1].feature_id != -1 else self.feature_id

            if self.types[current_feature_id] != FeaturesTypes.REAL:
                if features[current_feature_id] == self.unique_left and self.left:
                    path.append(self.left)
                    return self.left.aux_lookup(features, path)
                
                if features[current_feature_id] != self.unique_left and self.right:
                    path.append(self.right)
                    return self.right.aux_lookup(features, path)
                
            else:
                if features[current_feature_id] <= self.unique_left and self.left:
                    path.append(self.left)
                    return self.left.aux_lookup(features, path)
                
                if features[current_feature_id] > self.unique_left and self.right:
                    path.append(self.right)
                    return self.right.aux_lookup(features, path)
                

    def lookup_in_tree(self, features):
        path = [self]
        self.aux_lookup(features, path)
        return path


    def add_training_point(self, features: List[float], label: bool) -> None:
        path = Tree.lookup_in_tree(self, features)
        self.features = np.row_stack((self.features, features))
        self.labels = np.append(self.labels, label)

        for node in path:
            node.c += 1
            if node.c > node.beta * len(node.features):
                node.c = 0
                node = Tree(node.features, node.labels, node.types, c = node.c, h = node.current_h)


    def del_training_point(self, features: List[float], label: bool) -> None:
        path = Tree.lookup_in_tree(self, features)
        index_to_delete = np.where(np.all(self.features == features, axis=1))
        self.features = np.delete(self.features, index_to_delete, axis=0)
        self.labels = np.delete(self.labels, index_to_delete)

        for node in path:
            node.c += 1
            if node.c > node.beta * len(node.features):
                node.c = 0
                node = Tree(node.features, node.labels, node.types, c = node.c, h = node.current_h)
        