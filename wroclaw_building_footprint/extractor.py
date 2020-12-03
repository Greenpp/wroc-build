import math
from io import BytesIO
from typing import Iterable

from owslib.wms import WebMapService
from PIL import Image


def _calculate_bounding_box(
    center_lon: float,
    center_lat: float,
    side_length: int,
) -> Iterable[float]:
    """
    Calculate bounding box for WebMapService protocol
    """
    R = 6371000
    r = side_length / 2

    dLat = 360 * r / R
    dLon = dLat * math.cos(math.radians(center_lat))

    bottom_left_lon = center_lon - dLon
    bottom_left_lat = center_lat - dLat
    up_right_lon = center_lon + dLon
    up_right_lat = center_lat + dLat

    return bottom_left_lat, bottom_left_lon, up_right_lat, up_right_lon


class MapExtractor:
    def __init__(
        self,
        url: str = 'http://gis1.um.wroc.pl/arcgis/services/ogc/OGC_ortofoto_2018/MapServer/WMSServer?',
        area_size: int = 30,
    ) -> None:
        self.wms = WebMapService(url)

        self.area_size = area_size

    def get_area_at(self, lon: float, lat: float):
        """
        Extracts map fragment around a given point and return it as a PIL image
        """
        bbox = _calculate_bounding_box(lon, lat, self.area_size)

        response = self.wms.getmap(
            layers=['0'],
            srs='EPSG:4326',
            bbox=bbox,
            size=(512, 512),
            format='image/png',
        )

        img = Image.open(BytesIO(response.read()))

        return img
