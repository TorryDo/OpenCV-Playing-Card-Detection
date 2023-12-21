import math


class MathHelper:

    @staticmethod
    def length(x1, y1, x2, y2):
        """
        This function calculates the Euclidean distance between two points in a 2D plane.

        Args:
            x1: The x-coordinate of the first point.
            y1: The y-coordinate of the first point.
            x2: The x-coordinate of the second point.
            y2: The y-coordinate of the second point.

        Returns:
            The Euclidean distance between the two points.
        """

        # Use the distance formula to calculate the length
        length = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

        return length
