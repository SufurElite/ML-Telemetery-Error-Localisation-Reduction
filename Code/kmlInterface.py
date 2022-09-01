"""
    This file will deal with parsing the values from the Habitat Map in the data to better deal with and predict habitat covariates better.
    This will make use of the shapely library. 
"""

from utm import from_latlon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pykml import parser

class Habitat():
    """
        A simple Habitat class that can have multiple polygons per Habitat
    """
    def __init__(self, habitat_title: str):
        """
            Initialize empty lists for the Habitat's points and for its polygons
        """
        self.pointsList = []
        self.polygons = []
        self.habitat_title = habitat_title
    
    def addPolygon(self, points: list):
        """
            Add a new polygon to the habitats from the point list
        """
        self.pointsList.append(points)
        self.polygons.append(Polygon(points))
        
    def isInside(self, x, y):
        """
            Checks whether the x,y value is inside any of the habitats polygons
        """
        p = Point(x,y)
        for poly in self.polygons:
            if poly.contains(p): return True
        return False

    def getCoordinates(self):
        return self.pointsList

class HabitatMap():
    HABITAT_TYPES = ["bamboo","forest","high_paramo","open_paramo"]
    habitats = []
    def __init__(self, CUSTOM_TYPES=None):
        if CUSTOM_TYPES:
            self.HABITAT_TYPES = CUSTOM_TYPES
        for habitat in self.HABITAT_TYPES:
            self.habitats.append(self.loadPolygon(habitat))
    
    def loadPolygon(self, habitat_title):
        """
            For a given habitat title, load the associated kml data file,
            parse the lat,lng points, rewrite it in UTM values, and 
            store the generated Habitat object
        """
        currHabitat = Habitat(habitat_title)
        # parsing based on 
        # https://stackoverflow.com/questions/67454345/google-kml-file-to-python
        with open("../Data/Habitat Map/"+habitat_title+".kml", "r") as f:
            root = parser.parse(f).getroot()
            namespace = {"kml": 'http://www.opengis.net/kml/2.2'}
            pms = root.xpath(".//kml:Placemark[.//kml:Polygon]", namespaces=namespace)
        for poly in pms:
            """
                go through all polygon linear ring coordinates in the files; the long lat pairs 
                are separated by a ,0 (with one last ',0' at the end, which is why we delete)
            """
            points = []
            pts = str(poly.Polygon.outerBoundaryIs.LinearRing.coordinates).strip().split(",0")
            del pts[len(pts)-1]
            for pt in pts:
                lnglat = pt.split(",")
                assert(len(lnglat)==2)
                utmValues = from_latlon(float(lnglat[1]), float(lnglat[0]))
                points.append((utmValues[0], utmValues[1]))
            currHabitat.addPolygon(points)

        return currHabitat
    
    def whichHabitat(self, UTMx, UTMy):
        """
            Takes in a UTMx, UTMy value pair and determines which habitat
            the point lies within
        """
        for idx in range(len(self.habitats)):
            if self.habitats[i].isInside(UTMx, UTMy):
                return idx, self.HABITAT_TYPES[i]
        # if no value is found, return -1
        return -1

    def getHabitat(self, habitat_title) -> Habitat:
        """
            Takes in a habitat title and returns its associated Habitat object
        """
        return self.habitats[self.HABITAT_TYPES.index(habitat_title)]
    
    def getHabitats(self):
        return self.HABITAT_TYPES

if __name__=="__main__":
    habMap = HabitatMap()
    hab = habMap.getHabitat("bamboo")
    print(hab.getCoordinates())
    