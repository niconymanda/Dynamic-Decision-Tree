from typing import List, Tuple, Dict

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """

    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.min_split_points = 1 # To successfully run 2
        self.feature_id = None
        self.split_on = None

    
    def general_get_gini(labels: List[bool]):
        """Compute the Gini score for a set of labels. Please note this acts as a workaround to fulfill the requirements. 

        Parameters
        ----------
        labels: List[bool]
            The labels of the set.

        Returns
        -------
        float
            The Gini score for the set of labels.
        """
        number_true = sum(labels)
        prob_true = number_true / len(labels) 
        prob_false = (len(labels) - number_true) / len(labels) 

        gini_i = 1 - prob_true**2 - prob_false**2

        return gini_i


    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        gini_i = PointSet.general_get_gini(self.labels)

        return gini_i
    
        raise NotImplementedError('Please implement this function for Question 1')
    
    def associate_unique_class_to_label(self, feature_i: List[float], category_elem: float, continuous_indicator: int = None) -> Dict:
        """Associate unique classes to labels based on a feature. The labels are categorised differently depending on whether the features are continuous or not.

        Parameters
        ----------
        feature_i : List[float]
            The values of a specific feature.
        category_elem : float
            The unique value of the feature used for categorization.
        continuous_indicator : optional
            A flag indicating whether the feature is continuous or not. If None, the feature is considered non-continuous.

        Returns
        -------
        dict
            A dictionary mapping unique feature values to lists of corresponding labels.
        """        
        if continuous_indicator is None:
            labels_category = [self.labels[i] for i in list(np.where(feature_i == category_elem)[0])]
            other_labels = [self.labels[i] for i in list(np.where(feature_i != category_elem)[0])]
            label_dict = {category_elem: labels_category, "rest": other_labels}

        else:
            labels_category = [self.labels[i] for i in list(np.where(feature_i <= category_elem)[0])]
            other_labels = [self.labels[i] for i in list(np.where(feature_i > category_elem)[0])]
            label_dict = {category_elem: labels_category, "rest": other_labels}

        return label_dict
    

    def calculate_gini_split(self, label_dict: Dict) -> float:
        """Calculate the Gini score for splitting the set along a specific feature.

        Parameters
        ----------
        label_dict : dict
            A dictionary mapping unique feature values to lists of corresponding labels.

        Returns
        -------
        float
            The Gini score for splitting the set along the specified feature.
        """
        gini_weights = []
        ginis = []

        for key in label_dict.keys():
            gini_weights.append(len(label_dict[key])/len(self.labels))
            ginis.append(PointSet.general_get_gini(label_dict[key]))
        
        gini_split = sum(weight * gini for weight, gini in zip(gini_weights, ginis))

        return gini_split
    

    def get_max_gini(best_gini: float, best_feature_id: int, best_split_on: float,
                 current_gini: float, current_feature_id: int, current_split_on: float) -> Tuple[float, int, float]:
        """Update the best Gini score and associated feature and split information.

        Parameters
        ----------
        best_gini : float
            The current best Gini score.
        best_feature_id : int
            The current best feature ID.
        best_split_on : float
            The current best split value.
        current_gini : float
            The Gini score to compare with the current best Gini.
        current_feature_id : int
            The feature ID associated with the current Gini score.
        current_split_on : float
            The split value associated with the current Gini score.

        Returns
        -------
        Tuple[float, int, float]
            Updated best Gini score, feature ID, and split value.
        """

        if current_gini > best_gini:
            best_gini = current_gini
            best_feature_id = current_feature_id
            best_split_on = current_split_on

        return best_gini, best_feature_id, best_split_on
    

    def aux_best_gain_and_unique_left(self, feature_i: List[float], category_elem: float, best_gini: float, best_feature_id: int, best_split_on: float,
                                current_feature_id: int, current_split_on: float = None, continuous_indicator: int = None) -> Tuple[float, int, float]:                                
        """Auxiliary function for get_best_gain_and_unique_left().

        This function reduces the lines of code and enhances script readability. It calculates the Gini score for splitting
        the set along a specific feature, considering the minimum split points constraint.

        Parameters
        ----------
        feature_i : List[float]
            The values of a specific feature.
        category_elem : float
            The unique feature value.
        best_gini : float
            The current best Gini score.
        best_feature_id : int
            The current best feature ID.
        best_split_on : float
            The current best split value.
        current_feature_id : int
            The feature ID associated with the current Gini score.
        current_split_on : Optional[float], optional
            The split value associated with the current Gini score (default is None).
        continuous_indicator : Optional[bool], optional
            Indicator for continuous features (default is None).    

        Returns
        -------  
        Tuple[float, int, float]
            Updated best Gini score, feature ID, and split value.          
        """
        gini = PointSet.get_gini(self)

        label_dict = PointSet.associate_unique_class_to_label(self, feature_i, category_elem, continuous_indicator)

        contains_less_than_min_split = any(len(values) < self.min_split_points for values in label_dict.values())

        if not contains_less_than_min_split:
            current_gini = gini - PointSet.calculate_gini_split(self, label_dict)
            if continuous_indicator:
                category_elem = current_split_on
            best_gini, best_feature_id, best_split_on = PointSet.get_max_gini(best_gini, best_feature_id, best_split_on, current_gini, current_feature_id, category_elem)
        
        return best_gini, best_feature_id, best_split_on


    def get_best_gain_and_unique_left(self) -> Tuple[int, float]:
        """Find the feature along which splitting provides the best gain, considering the minimum split points constraint.

        Returns
        -------
        Tuple[int, float]
            The ID of the feature along which splitting the set provides the best Gini gain, and the best Gini gain achievable by splitting this set along one of its features.
        """
        best_gini = -1
        best_feature_id = -1
        best_split_on = -1
        # gini = PointSet.get_gini(self)

        for i, feature in enumerate(self.features.T):
            unique_feature_vals = np.unique(feature)

            if self.types[i] == FeaturesTypes.BOOLEAN or (self.types[i] == FeaturesTypes.CLASSES and len(unique_feature_vals) < 3):
                category_elem = min(unique_feature_vals)

                best_gini, best_feature_id, best_split_on = PointSet.aux_best_gain_and_unique_left(self, feature, category_elem, best_gini, best_feature_id, best_split_on, i)
            
            elif self.types[i] == FeaturesTypes.CLASSES:   
                tmp_best_gini = -1
                tmp_best_feature_id = -1
                tmp_best_split_on = -1

                for j, category_elem in enumerate(unique_feature_vals):
                    tmp_best_gini, tmp_best_feature_id, tmp_best_split_on = PointSet.aux_best_gain_and_unique_left(self, feature, category_elem, tmp_best_gini, tmp_best_feature_id, tmp_best_split_on, j)

                best_gini, best_feature_id, best_split_on = PointSet.get_max_gini(best_gini, best_feature_id, best_split_on, tmp_best_gini, i, tmp_best_split_on)

            else:
                tmp_best_gini = -1
                tmp_best_feature_id = -1
                tmp_best_split_on = -1
                sorted_feature = np.sort(unique_feature_vals)

                for j, category_elem in enumerate(sorted_feature[:-1]):
                    current_split_on = (category_elem+sorted_feature[j+1])/2
                    tmp_best_gini, tmp_best_feature_id, tmp_best_split_on = PointSet.aux_best_gain_and_unique_left(self, feature, category_elem, tmp_best_gini, tmp_best_feature_id, tmp_best_split_on, j, current_split_on, 1)
                
                best_gini, best_feature_id, best_split_on = PointSet.get_max_gini(best_gini, best_feature_id, best_split_on, tmp_best_gini, i, tmp_best_split_on)

        self.feature_id = best_feature_id
        self.split_on = best_split_on
    
        return self.feature_id, best_gini, self.split_on
    

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """

        best_feature_id, best_gini, _ = PointSet.get_best_gain_and_unique_left(self)
        return best_feature_id, best_gini

        raise NotImplementedError('Please implement this methode for Question 2')
    

    def split_node(self):
        """Split the set of points along the feature that provides the best gain. This is used for the constructor in the Tree.py file.

        Returns
        -------
        Tuple[List[List[float]], List[List[float]], List[bool], List[bool], float, int]
            A tuple containing the features and labels for the left and right subsets,
            the unique left feature value, and the ID of the feature providing the best gain.
        """
        self.features = np.array(self.features, dtype=object)
        features_t = self.features.T

        best_feature_id, _, unique_left = PointSet.get_best_gain_and_unique_left(self) # _ = best gini value
        best_feature_vector = features_t[best_feature_id]
        
        if self.types[best_feature_id] == FeaturesTypes.BOOLEAN or self.types[best_feature_id] == FeaturesTypes.CLASSES:
            unique_right = [x for x in list(set(best_feature_vector)) if x != unique_left]
            indices_left = [i for i, x in enumerate(best_feature_vector) if x == unique_left]
            indices_right = [i for i, x in enumerate(best_feature_vector) if x in unique_right]
        else: 
            unique_right = [x for x in list(set(best_feature_vector)) if x > self.split_on]
            indices_left = [i for i, x in enumerate(best_feature_vector) if x <= self.split_on]
            indices_right = [i for i, x in enumerate(best_feature_vector) if x > self.split_on]

        # new_features = np.delete(self.features, best_feature_id, 1)
        features_left = np.array([[sublist[index] for index in indices_left] for sublist in features_t], dtype=object).T
        features_right = np.array([[sublist[index] for index in indices_right] for sublist in features_t], dtype=object).T

        labels_left = [self.labels[index] for index in indices_left]
        labels_right = [self.labels[index] for index in indices_right]

        return features_left, features_right, labels_left, labels_right, unique_left, unique_right, best_feature_id


    def get_best_threshold(self) -> float:
        """Get the best threshold for splitting based on the feature type.

        Returns
        -------
        float
            The best threshold for splitting. If the feature type is BOOLEAN, returns None.
        """

        if self.feature_id is None:
            raise ValueError("get_best_gain has not been called.")

        type_split_feature = self.types[self.feature_id]

        if type_split_feature == FeaturesTypes.BOOLEAN:
            return None
        else: 
            return self.split_on
    

    # Optimisation:
    # Initially, I had saved all of the ginis and feature IDs as lists and then chose the max. Instead, I used comparison to determine which gini was the so-far best.
    # - remove features that all contain the same value. For these the gini is computed. Unnecessary.
    # - compute gini incrementally

