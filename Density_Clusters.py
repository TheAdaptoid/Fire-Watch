class Cluster:
    """
    A class representing a cluster of vectors.

    Attributes
    ----------
    name : str
        The name of the cluster.

    vectors : list[list[int|float]]
        A list of vectors assigned to the cluster.

    silhouetteScore : float
        The silhouette score of the cluster.
    """

    def __init__(self, name: str):
        """
        Initialize a cluster with a name and centroid.
        
        Parameters
        ----------
        name : str
            The name of the cluster.
        """

        self.name = name
        self.vectors: list[list[int|float]] = []
        self.silhouetteScore: float = 0 # TODO: Verify if this is actually used

class Density_Clusters:
    def __init__(self, matrix: list[list[int|float]], maxEpochs: int = 100):
        self.matrix = matrix
        self.maxEpochs = maxEpochs
        pass

    def Train(self) -> int:
        pass

    def Predict(self) -> Cluster:
        pass