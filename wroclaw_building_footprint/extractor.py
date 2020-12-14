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
    """Calculate a square bounding box around given coordinates

    Args:
        center_lat: Center latitude to calculate bounding box around.
        center_lon: Center longitude to calculate bounding box around.
        side_length: Bounding box side length.

    Returns:
        Four floats, bottom-left longitude, bottom-left latitude, upper-right longitude, upper-right latitude.
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
    """Class used to extract orthophoto map fragments."""

    def __init__(
        self,
        url: str = 'http://gis1.um.wroc.pl/arcgis/services/ogc/OGC_ortofoto_2018/MapServer/WMSServer?',
    ) -> None:
        self.wms = WebMapService(url)

    def get_area_at(self, lon: float, lat: float, area_size: float):
        """Create an image of the specified area.

        Args:
            lon: Center longitude of the area.
            lat: Center latitude of the area.
            area_size: Side length of the area.

        Returns:
            An PIL Image orthophoto of the selected area.
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
