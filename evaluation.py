from typing import List

def calculate_confusion_matrix(expected_results: List[bool], actual_results: List[bool]):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, elem in enumerate(expected_results):
        
        if elem and actual_results[index]:
            TP += 1
        elif not elem and not actual_results[index]:
            TN += 1
        elif not elem and actual_results[index]:
            FP += 1
        else: 
            FN += 1

    return TP, TN, FP, FN

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """

    TP, TN, FP, FN = calculate_confusion_matrix(expected_results, actual_results)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

    raise NotImplementedError('Implement this method for Question 3')

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """

    precision, recall = precision_recall(expected_results, actual_results)

    f_score = (2*precision * recall) / (recall + precision)

    return f_score

    raise NotImplementedError('Implement this method for Question 3')
