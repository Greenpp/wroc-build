import math
from io import BytesIO
from typing import Iterable

from owslib.wms import WebMapService
from PIL import Image


def _calculate_bounding_box(
    center_lat: float,
    center_lon: float,
    side_length: float,
) -> Iterable[float]:
    """
    Calculate bounding box for WebMapService protocol
    """
    l = side_length / 2

    rad_lat = math.radians(center_lat)

    dLat = l / (
        111132.92
        - 559.82 * math.cos(2 * rad_lat)
        + 1.175 * math.cos(4 * rad_lat)
        - 0.0023 * math.cos(6 * rad_lat)
    )
    dLon = l / (
        111412.84 * math.cos(rad_lat)
        - 93.5 * math.cos(3 * rad_lat)
        + 0.118 * math.cos(5 * rad_lat)
    )

    bottom_left_lon = center_lon - dLon
    bottom_left_lat = center_lat - dLat
    up_right_lon = center_lon + dLon
    up_right_lat = center_lat + dLat

    return bottom_left_lon, bottom_left_lat, up_right_lon, up_right_lat


class MapExtractor:
    def __init__(
        self,
        url: str = 'http://gis1.um.wroc.pl/arcgis/services/ogc/OGC_ortofoto_2018/MapServer/WMSServer?',
    ) -> None:
        self.wms = WebMapService(url)

    def get_area_at(self, lon: float, lat: float, area_size: float):
        """
        Extracts map fragment around a given point and return it as a PIL image
        """
        bbox = _calculate_bounding_box(lon, lat, area_size)

        response = self.wms.getmap(
            layers=['0'],
            srs='EPSG:4326',
            bbox=bbox,
            size=(512, 512),
            format='image/png',
        )

        img = Image.open(BytesIO(response.read()))

        return img
