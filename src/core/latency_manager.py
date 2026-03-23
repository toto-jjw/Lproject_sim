"""Communication Latency Manager for Lunar Rover Simulation

달 탐사 통신 지연 시뮬레이션:
- 지구-달 통신 지연: ~1.3초 (round-trip)
- 신호 손실: 확률적 dropout
"""

import time
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LatencyConfig:
    """Latency 시뮬레이션 설정"""
    delay_seconds: float = 1.3      # 통신 지연 (초)
    dropout_rate: float = 0.05     # 패킷 손실 확률 (0~1)
    max_buffer_size: int = 100     # 버퍼 최대 크기


class LatencyManager:
    """
    통신 지연 및 신호 손실 시뮬레이터
    
    주요 기능:
    - 설정 가능한 지연 시간
    - 확률적 패킷 손실
    - FIFO 버퍼 관리
    """
    
    def __init__(self, delay_seconds: float = 1.3, dropout_rate: float = 0.05, 
                 max_range: float = 100.0):  # max_range는 하위 호환성용
        """
        Args:
            delay_seconds: 통신 지연 시간 (초)
            dropout_rate: 패킷 손실 확률 (0~1)
            max_range: 미사용 (하위 호환성)
        """
        self.delay_seconds = delay_seconds
        self.dropout_rate = dropout_rate
        self.buffer = deque(maxlen=100)  # (timestamp, data)
        
        # 통계
        self._packets_sent = 0
        self._packets_dropped = 0
        
    def send(self, data: Any, sender_pos: tuple = (0, 0, 0)) -> bool:
        """
        데이터를 버퍼에 추가 (지연 후 수신 가능)
        
        Args:
            data: 전송할 데이터
            sender_pos: 미사용 (하위 호환성)
            
        Returns:
            True: 전송 성공, False: 드롭됨
        """
        self._packets_sent += 1
        
        # 랜덤 드롭아웃
        if random.random() < self.dropout_rate:
            self._packets_dropped += 1
            return False
            
        self.buffer.append((time.time(), data))
        return True
        
    def receive(self) -> Optional[Any]:
        """
        지연 시간이 지난 데이터 하나 수신
        
        Returns:
            지연 시간이 지난 데이터 또는 None
        """
        if not self.buffer:
            return None
            
        timestamp, data = self.buffer[0]
        if time.time() - timestamp >= self.delay_seconds:
            self.buffer.popleft()
            return data
        return None
        
    def get_latest(self) -> Optional[Any]:
        """
        준비된 가장 최신 데이터 반환 (오래된 것들은 버림)
        
        Returns:
            가장 최신 데이터 또는 None
        """
        latest_data = None
        while True:
            data = self.receive()
            if data is None:
                break
            latest_data = data
        return latest_data
    
    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()
        
    def get_statistics(self) -> dict:
        """통계 정보 반환"""
        return {
            "packets_sent": self._packets_sent,
            "packets_dropped": self._packets_dropped,
            "drop_rate": self._packets_dropped / max(1, self._packets_sent),
            "buffer_size": len(self.buffer)
        }
