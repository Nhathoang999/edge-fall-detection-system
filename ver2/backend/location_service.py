import asyncio
import logging
from typing import Optional, Tuple

try:
    from winsdk.windows.devices.geolocation import Geolocator, GeolocationAccessStatus
except ImportError:
    Geolocator = None
    GeolocationAccessStatus = None

logger = logging.getLogger("location_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class LocationService:
    """
    Module tự động trích xuất vị trí địa lý của máy tính (Windows 10/11) 
    và tạo URL Google Maps để đính kèm vào tin nhắn cảnh báo.
    """
    def __init__(self):
        # Khởi tạo đối tượng Geolocator của Windows API
        self.geolocator = Geolocator() if Geolocator else None

    async def get_current_location(self) -> Optional[Tuple[float, float]]:
        """
        Gọi Windows Geolocation API để lấy vĩ độ và kinh độ.
        Sử dụng Async/Await để không block luồng.
        """
        if not self.geolocator:
            logger.error("[LOCATION] Module winsdk chưa được cài đặt hoặc không hỗ trợ. Vui lòng chạy: pip install winsdk")
            return None
            
        try:
            # Yêu cầu người dùng / hệ thống cấp quyền truy cập vị trí
            access_status = await Geolocator.request_access_async()
            
            if access_status != GeolocationAccessStatus.ALLOWED:
                logger.warning("[LOCATION] Quyền truy cập vị trí bị từ chối hoặc bị tắt trong cài đặt Windows.")
                logger.warning("👉 HƯỚNG DẪN BẬT: Settings → Privacy & Security → Location → Turn On.")
                return None
                
            # Lấy vị trí địa lý hiện tại
            pos = await self.geolocator.get_geoposition_async()
            
            if pos and pos.coordinate:
                lat = pos.coordinate.latitude
                lon = pos.coordinate.longitude
                logger.info(f"[LOCATION] Đã lấy thành công vị trí: {lat}, {lon}")
                return (lat, lon)
            else:
                logger.error("[LOCATION] Không thể đọc được tọa độ từ thiết bị định vị.")
                return None
                
        except Exception as e:
            logger.error(f"[LOCATION] Đã xảy ra lỗi khi gọi Geolocation API: {e}")
            return None

    def get_google_maps_link(self, lat: float, lon: float) -> str:
        """
        Tạo đường link Google Maps từ tọa độ.
        """
        return f"https://maps.google.com/?q={lat},{lon}"

# Singleton instance để dùng chung trong dự án
location_service = LocationService()
