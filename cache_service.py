"""
Redis 缓存服务层
提供声明式缓存装饰器，加速高频查询
Redis 不可用时自动降级为内存缓存（DictCache）
"""

import os
import json
import hashlib
import time
import functools
from threading import Lock

# Redis 配置
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# 全局缓存实例
_cache_backend = None
_redis_available = False
_cache_lock = Lock()


class DictCache:
    """内存字典缓存（Redis 不可用时的降级方案）"""

    def __init__(self, default_ttl=300):
        self.store = {}  # {key: (value, expiry_time)}
        self.default_ttl = default_ttl
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            if key in self.store:
                value, expiry = self.store[key]
                if time.time() < expiry:
                    return value
                else:
                    del self.store[key]
            return None

    def set(self, key, value, ttl=None):
        ttl = ttl or self.default_ttl
        with self._lock:
            self.store[key] = (value, time.time() + ttl)

    def delete(self, key):
        with self._lock:
            self.store.pop(key, None)

    def clear(self):
        with self._lock:
            self.store.clear()

    def exists(self, key):
        return self.get(key) is not None

    def stats(self):
        with self._lock:
            now = time.time()
            valid = sum(1 for _, exp in self.store.values() if now < exp)
            expired = len(self.store) - valid
            return {'total_keys': len(self.store), 'valid': valid, 'expired': expired}


class RedisCache:
    """Redis 缓存封装"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=REDIS_DB):
        self.client = None
        self.connected = False
        self._connect(host, port, password, db)

    def _connect(self, host, port, password, db):
        try:
            import redis
            self.client = redis.Redis(
                host=host, port=port, password=password, db=db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # 测试连接
            self.client.ping()
            self.connected = True
            print(f"[Cache] Redis 连接成功: {host}:{port}/{db}")
        except ImportError:
            print("[Cache] redis 库未安装，使用内存缓存")
        except Exception as e:
            print(f"[Cache] Redis 连接失败 ({e})，使用内存缓存")

    def get(self, key):
        if not self.connected:
            return None
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except:
            return None

    def set(self, key, value, ttl=300):
        if not self.connected:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            return self.client.setex(key, ttl, serialized)
        except:
            return False

    def delete(self, key):
        if not self.connected:
            return False
        try:
            return self.client.delete(key) > 0
        except:
            return False

    def clear(self):
        if not self.connected:
            return
        try:
            self.client.flushdb()
        except:
            pass

    def exists(self, key):
        if not self.connected:
            return False
        try:
            return self.client.exists(key) > 0
        except:
            return False

    def stats(self):
        if not self.connected:
            return {'status': 'disconnected'}
        try:
            info = self.client.info()
            return {
                'status': 'connected',
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'connected_clients': info.get('connected_clients', 0)
            }
        except:
            return {'status': 'error'}


def get_cache_backend():
    """获取缓存后端（单例模式）"""
    global _cache_backend, _redis_available

    with _cache_lock:
        if _cache_backend is None:
            # 先尝试 Redis
            redis_cache = RedisCache()
            if redis_cache.connected:
                _cache_backend = redis_cache
                _redis_available = True
            else:
                _cache_backend = DictCache(default_ttl=300)
                _redis_available = False

        return _cache_backend


def cached(ttl=300, prefix=''):
    """
    缓存装饰器

    用法:
        @cached(ttl=60, prefix='stats')
        def get_statistics():
            ... 昂贵操作 ...

    参数:
        ttl: 缓存过期时间（秒），默认5分钟
        prefix: 缓存键前缀，用于分组管理
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_backend()

            # 生成缓存键
            key_parts = [prefix, func.__name__]

            # 将参数纳入键（仅基本类型）
            for a in args[1:]:  # 跳过 self
                if isinstance(a, (str, int, float, bool)):
                    key_parts.append(str(a))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")

            key_string = ':'.join(key_parts)
            cache_key = f"hotel:{hashlib.md5(key_string.encode()).hexdigest()}"

            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 执行原函数
            result = func(*args, **kwargs)

            # 写入缓存
            cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


def invalidate_cache(pattern=None):
    """
    使缓存失效
    pattern: 键模式（如 'hotel:*' 清除所有酒店相关缓存）
             如果为 None，清除全部
    """
    cache = get_cache_backend()

    if isinstance(cache, RedisCache) and pattern:
        try:
            keys = cache.client.keys(pattern)
            if keys:
                cache.client.delete(*keys)
            return {'invalidated': len(keys), 'pattern': pattern}
        except:
            pass
    else:
        cache.clear()
        return {'invalidated': 'all'}


def get_cache_stats():
    """获取缓存统计信息"""
    cache = get_cache_backend()
    return {
        'backend': 'redis' if _redis_available else 'memory',
        'available': _redis_available,
        **cache.stats()
    }
