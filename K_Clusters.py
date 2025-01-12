from random import choices
from typing import Callable

class Distance_Functions:
    def _Verify_Dimensionality(self, aVector: list[int|float], bVector: list[int|float]) -> None:
        """
        Verifies that two vectors have the same dimensionality.

        Parameters
        ----------
        aVector : list[ int | float ]
            The first vector.
        bVector : list[ int | float ]
            The second vector.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """

        if len(aVector) != len(bVector):
            raise ValueError(f"Vectors must have the same dimensionality: {len(aVector)} != {len(bVector)}")

    def Euclidean_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculates the Euclidean distance between two vectors.

        Parameters
        ----------
        aVector : list[int|float]
            The first vector.
        bVector : list[int|float]
            The second vector.

        Returns
        -------
        float
            The Euclidean distance between the two vectors.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """

        self._Verify_Dimensionality(aVector, bVector)

        distance: float = 0.0

        for i in range(len(aVector)):
            distance += (aVector[i] - bVector[i]) ** 2

        return distance ** 0.5
    
    def Cosine_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculates the cosine distance between two vectors.

        Parameters
        ----------
        aVector : list[ int | float ]
            The first vector.
        bVector : list[ int | float ]
            The second vector.

        Returns
        -------
        float
            The Cosine distance between the two vectors. Ranges from 0 to 1.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """

        self._Verify_Dimensionality(aVector, bVector)

        dotProduct: float = sum(a * b for a, b in zip(aVector, bVector))
        magnitudeA: float = sum(a * a for a in aVector) ** 0.5
        magnitudeB: float = sum(b * b for b in bVector) ** 0.5

        denominator: float = max(0.001, magnitudeA * magnitudeB) # Prevents division by zero

        return 1 - (dotProduct / denominator)
    
    def Manhattan_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculates the Manhattan distance between two vectors.

        Parameters
        ----------
        aVector : list[int|float]
            The first vector.
        bVector : list[int|float]
            The second vector.

        Returns
        -------
        float
            The Manhattan distance between the two vectors.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """

        self._Verify_Dimensionality(aVector, bVector)
        
        distance: float = 0.0

        for i in range(len(aVector)):
            distance += abs(aVector[i] - bVector[i])

        return distance

class Cluster:
    """
    A class representing a cluster of vectors.

    Attributes
    ----------
    name : str
        The name of the cluster.

    centroid : list[int|float]
        A vector that represents the central point of the cluster.

    vectors : list[list[int|float]]
        A list of vectors assigned to the cluster.

    silhouetteScore : float
        The silhouette score of the cluster.


    Methods
    -------
    Recalculate_Centroid() -> bool
        Recalculate the centroid of the cluster.
    """

    def __init__(self, name: str, centroid: list[int|float] = []):
        """
        Initialize a cluster with a name and centroid.
        
        Parameters
        ----------
        name : str
            The name of the cluster.

        centroid : list[int|float], optional\n
            A vector that represents the central point of the cluster.\n
            (Default is an empty list).
        """

        self.name = name
        self.centroid = centroid
        self.vectors: list[list[int|float]] = []
        self.silhouetteScore: float = 0

    def _Mean_Vector(self, vectors: list[list[int|float]]) -> list[int|float]:
        """
        Calculate the average vector from a list of vectors.

        Parameters
        ----------
        vectors : list[list[int|float]]
            A list of vectors to average.

        Returns
        -------
        list[int|float]
            The average vector.
        """

        dimensionality: int = len(vectors[0])
        sumVector: list[int|float] = [0] * dimensionality

        for vector in vectors:
            for i in range(dimensionality):
                sumVector[i] += vector[i]

        for i in range(dimensionality):
            sumVector[i] = round((sumVector[i] / len(vectors)), 4)

        return sumVector
    
    def _Median_Vector(self, vectors: list[list[int|float]], distanceFunction: Callable) -> list[int|float]:
        """
        Calculate the median vector from a list of vectors.

        Parameters
        ----------
        vectors : list[list[int|float]]
            A list of vectors to calculate the median of.
        distanceFunction : Callable
            A function to calculate the distance between two vectors.

        Returns
        -------
        list[int|float]
            The median vector.
        """

        distanceSums: list[float] = [0 for _ in range(len(vectors))]

        # Calucate the total distance to all other vectors for each vector
        for i, iVector in enumerate(vectors):
            for jVector in vectors:
                distanceSums[i] += distanceFunction(aVector=iVector, bVector=jVector)

        minDistance: float = float('inf')
        medianVector: list[int|float] = []

        # Find the median vector
        for i in range(len(vectors)):
            if distanceSums[i] < minDistance:
                minDistance = distanceSums[i]
                medianVector = vectors[i]

        return medianVector

    def Recalculate_Centroid(self, centroidStrategy: str, distanceFunction: Callable) -> bool:
        """
        Recalculate the centroid of the cluster according to the centroid strategy. If there are no vectors, return True.

        Parameters
        ----------
        centroidStrategy : str
            The strategy to use for recalculating the centroid.
        distanceFunction : Callable
            The distance function to use for calculating the distance between vectors for median centroids.

        Returns
        -------
        bool
            True if the centroid did not change, False if it did.
        """

        # If there are no vectors, return the unchanged centroid
        if self.vectors == []: return True

        # Keep track of the previous centroid
        previousCentroid: list[int|float] = self.centroid

        # Recalculate the centroid
        if centroidStrategy == "median":
            self.centroid = self._Median_Vector(self.vectors, distanceFunction)
        else:
            self.centroid = self._Mean_Vector(self.vectors)

        # Check if the centroid changed
        return previousCentroid == self.centroid
    
    def __dict__(self) -> dict:
        """
        Convert the cluster object to a dictionary.

        Returns
        -------
        dict
            A dictionary with keys "name", "centroid", and "vectors".
        """

        return {
            "name": self.name,
            "centroid": self.centroid,
            "vectors": self.vectors
        }
    
    def __str__(self) -> str:
        """
        Convert the cluster object to a string.

        Returns
        -------
        str
            A string representation of the cluster.
        """

        return str(self.__dict__())
    
class K_Clusters:
    """
    K-Clusters model for clustering vectors with either mean or median centroids.

    Attributes
    ----------
    trainingMatrix : list[ list[ int | float ] ]
        The matrix of vectors to train on.
    
    kGroups : int
        The number of clusters to group the data into.
    
    distanceStrategy : str
        The strategy to use for assigning vectors to clusters.

    centroidStrategy : str
        The strategy to use for recalculating the centroid of the cluster.
    
    maxEpochs : int
        The maximum number of epochs to train for.

    clusters : list[ Cluster ]
        A list of clusters.

    silhouetteScore : float
        The average silhouette score of the clusters.

    Methods
    -------
    Train() : int
        Train the model.

    Predict(vector: list[int|float])
        Predict the cluster for a given vector.
    """

    def __init__(self,
            trainingMatrix:list[list[int|float]], 
            kGroups: int = 3,
            distanceStrategy: str = "euclidean",
            centroidStrategy: str = "mean",
            maxEpochs: int = 100
        ) -> None:
        """
        Initialize a K-Clusters model with a training matrix and hyperparameters.

        Parameters
        ----------
        trainingMatrix : list[ list[ int | float ] ]
            The matrix of vectors to train on.
        
        kGroups : int, optional
            The number of clusters to group the data into. (Default is 3).
        
        distanceStrategy : str, optional
            The strategy to use for assigning vectors to clusters.\n
            "euclidean": Assign vectors to the cluster with the lowest *Euclidean* distance.\n
            "cosine": Assign vectors to the cluster with the highest *Cosine* similarity.\n
            "manhattan": Assign vectors to the cluster with the lowest *Manhattan* distance.\n
            (Default is "euclidean").

        centroidStrategy : str, optional
            The strategy to use for recalculating the centroid of a cluster.\n
            "mean": Assign centroids to the average of a given cluster's vectors. Computes in $O(n)$ time.\n
            "median":  Assign centroids to the median of a given cluster's vectors. Performs better with non-euclidean distance metrics. More resistant to outliers, but at a cost of increased run time, $O(n^2)$.\n
            (Default is "mean").
        
        maxEpochs : int, optional
            The maximum number of epochs to train for. (Default is 100).
        """

        self.trainingMatrix: list[list[int|float]] = trainingMatrix
        self.kGroups: int = kGroups
        self.distanceStrategy = distanceStrategy.lower()
        self.centroidStrategy = centroidStrategy.lower()
        self.maxEpochs: int = maxEpochs
        self.clusters: list[Cluster] = []

        self._Verify_Distance_Strategy()
        self._Verify_Centroid_Strategy()
        self.clusters = self._Initialize_Clusters()
        self.silhouetteScore: float = 0
    
    def _Initialize_Clusters(self) -> list[Cluster]:
        """
        Initialize clusters with random vectors from the training matrix.

        Returns
        -------
        list[Cluster]
            A list of initialized clusters.
        """

        clusters: list[Cluster] = []
        randomVectors: list[list[int|float]] = choices(self.trainingMatrix, k=self.kGroups)

        for i in range(self.kGroups):
            clusters.append(
                Cluster(
                    name=f"cluster_{i}",
                    centroid=randomVectors[i]
                )
            )

        return clusters
    
    def _Verify_Distance_Strategy(self) -> None:
        """
        Verifies that the distance strategy parameter is either "euclidean", "cosine", or "manhattan".
        """

        if self.distanceStrategy not in ["euclidean", "cosine", "manhattan"]:
            self.distanceStrategy = "euclidean"
    
    def _Verify_Centroid_Strategy(self) -> None:
        """
        Verifies that the centroid strategy parameter is either "mean" or "median".
        """

        if self.centroidStrategy not in ["mean", "median"]:
            self.centroidStrategy = "mean"

    def _Get_Distance_Function(self) -> Callable[[list[int|float], list[int|float]], float]:
        """
        Returns a distance function based on the strategy parameter.

        Returns
        -------
        Callable[[list[int|float], list[int|float]], float]
            A distance function corresponding to the strategy parameter.
        """

        if self.distanceStrategy == "euclidean":
            return Distance_Functions().Euclidean_Between
        elif self.distanceStrategy == "cosine":
            return Distance_Functions().Cosine_Between
        elif self.distanceStrategy == "manhattan":
            return Distance_Functions().Manhattan_Between

    def _Get_Nearest_Cluster(self, vector: list[int|float]) -> Cluster:
        """
        Finds the cluster with minimal separation from a given vector.

        Parameters
        ----------
        vector : list[ int | float ]
            The vector to find the nearest cluster for.

        Returns
        -------
        Cluster
            The cluster with minimal separation from the given vector.
        """

        distanceFunction: Callable = self._Get_Distance_Function()

        # Calculate centroid separations
        centroidSeparations: dict[int, float] = {
            centroid: distanceFunction(vector, cluster.centroid) 
            for centroid, cluster in enumerate(self.clusters)
        }

        # Find the index of the cluster with minimal separation
        indexOfNearestCluster: int = min(centroidSeparations, key=centroidSeparations.get)

        return self.clusters[indexOfNearestCluster]

    def Train(self) -> int:
        """
        Trains the K-Clusters model.

        Returns
        -------
        int
            The number of epochs until the model converged.
        """

        # Initialize variables
        epoch: int = 0
        isConverged: bool = False

        # Train
        while not isConverged and epoch < self.maxEpochs:
            # Clear the clusters
            for cluster in self.clusters: cluster.vectors = []

            # Assign vectors to clusters
            for vector in self.trainingMatrix:
                
                # Add the vector to the centroid of minimal separation
                self._Get_Nearest_Cluster(vector).vectors.append(vector)

            # Calculate new centroids
            convergenceArray: list[bool] = [
                cluster.Recalculate_Centroid(
                    centroidStrategy=self.centroidStrategy,
                    distanceFunction=self._Get_Distance_Function()
                ) for cluster in self.clusters
            ]

            # check for convergence
            isConverged = all(convergenceArray)

            # Increment epoch
            epoch += 1

        # Calculate silhouette scores
        self._Calculate_Silhouette_Scores()

        return epoch
    
    def Predict(self, vector: list[int|float]) -> Cluster:
        """
        Predicts the cluster to which a given vector belongs.

        Parameters
        ----------
        vector : list[ int | float ]
            The vector to predict the cluster for.

        Returns
        -------
        Cluster
            The predicted cluster the vector belongs to.
        """

        return self._Get_Nearest_Cluster(vector)
    
    def _Calculate_Silhouette_Scores(self) -> None:
        for cIndex, cluster in enumerate(self.clusters):
            silhouetteScores: list[float] = []

            for vector in cluster.vectors:
                # Get the distance between the vector and its centroid
                vectorToCentroid: float = self._Get_Distance_Function()(vector, cluster.centroid)

                # Get the distance between the vector and its nearest neighbor
                otherCentroids: list[list[int|float]] = [
                    cluster.centroid for cJndex, cluster in enumerate(self.clusters) if cIndex != cJndex
                ]
                vectorToNeighbor: float = min([self._Get_Distance_Function()(vector, centroid) for centroid in otherCentroids])

                # Calculate the silhouette score
                silhouetteScore: float = (vectorToNeighbor - vectorToCentroid) / max(vectorToNeighbor, vectorToCentroid)
                silhouetteScores.append(silhouetteScore)

            # Calculate the average silhouette score
            avgSilhouetteScore: float = sum(silhouetteScores) / len(silhouetteScores)
            cluster.silhouetteScore = round(avgSilhouetteScore, 4)

        self.silhouetteScore = round(
            number=sum([cluster.silhouetteScore for cluster in self.clusters]) / len(self.clusters),
            ndigits=4
        )

    def __dict__(self) -> dict:
        """
        Converts the parameters of the K-Clusters model into a dictionary.

        Returns
        -------
        dict
            A dictionary with keys "trainingMatrix", "kGroups", "strategy", "centroidStrategy", "maxEpochs", "clusters".
        """

        return {
            "trainingMatrix": self.trainingMatrix,
            "kGroups": self.kGroups,
            "strategy": self.distanceStrategy,
            "centroidStrategy": self.centroidStrategy,
            "maxEpochs": self.maxEpochs,
            "clusters": [cluster.__dict__() for cluster in self.clusters]
        }
    
    def __str__(self) -> str:
        """
        Converts the parameters of the K-Clusters model into a string.

        Returns
        -------
        str
            A string representation of the K-Clusters model.
        """

        return str(self.__dict__())