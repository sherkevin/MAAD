from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import numpy as np
from datetime import datetime

@dataclass
class AgentState:
    """智能体状态数据结构"""
    agent_id: str
    status: str  # active, idle, error
    last_update: datetime
    confidence_score: float
    processing_time: float
    memory_usage: float
    error_count: int = 0
    
    def update(self, **kwargs):
        """更新状态信息"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.last_update = datetime.now()

class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.state = AgentState(
            agent_id=agent_id,
            status="idle",
            last_update=datetime.now(),
            confidence_score=0.0,
            processing_time=0.0,
            memory_usage=0.0
        )
        self.memory = {}
        self.communication_history = []
    
    @abstractmethod
    def process_data(self, data: torch.Tensor) -> Dict[str, Any]:
        """处理数据，子类必须实现"""
        pass
    
    def communicate(self, message: Dict[str, Any], target_agent: str) -> bool:
        """智能体间通信"""
        try:
            self.communication_history.append({
                'timestamp': datetime.now(),
                'target': target_agent,
                'message': message
            })
            return True
        except Exception as e:
            self.state.error_count += 1
            return False
    
    def get_state(self) -> AgentState:
        """获取智能体状态"""
        return self.state
    
    def reset(self):
        """重置智能体状态"""
        self.state = AgentState(
            agent_id=self.agent_id,
            status="idle",
            last_update=datetime.now(),
            confidence_score=0.0,
            processing_time=0.0,
            memory_usage=0.0
        )
        self.memory.clear()
        self.communication_history.clear()
    
    def get_memory_usage(self) -> float:
        """获取内存使用量"""
        if hasattr(self, 'memory'):
            return len(str(self.memory)) / 1024.0  # KB
        return 0.0
