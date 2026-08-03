# my_logger.py (新增 hide_prefix 选项)

import logging
import sys

# --- LOGGER定义和通用的日志封装函数 (_log_message, info, debug, etc.) 保持不变 ---

LOGGER = logging.getLogger('GlobalAppLogger')
LOGGER.setLevel(logging.DEBUG)
_LOGGING_SETUP_COMPLETE = False


def setup_logging(log_file_path, level=logging.INFO, hide_prefix=False):
    """
    配置全局Logger，新增 hide_prefix 参数以控制是否隐藏日志前缀。

    :param log_file_path: 日志文件路径
    :param level: 最低日志级别
    :param hide_prefix: 如果为 True，则只输出原始消息，隐藏时间戳、级别等前缀。
    """
    global _LOGGING_SETUP_COMPLETE
    if _LOGGING_SETUP_COMPLETE:
        return
    if LOGGER.hasHandlers():
        LOGGER.handlers.clear()

    # --- 1. 定义 Formatters ---
    
    # 详细格式 (默认)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 简洁格式 (只输出消息)
    simple_formatter = logging.Formatter('%(message)s')

    # 2. 根据选项选择格式器
    if hide_prefix:
        formatter = simple_formatter
    else:
        formatter = detailed_formatter
        
    # --- 3. 配置 Handlers ---
    
    # 文件 Handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter) # <-- 应用选定的格式器
    LOGGER.addHandler(file_handler)

    # Stream Handler (控制台)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter) # <-- 应用选定的格式器
    # 控制台通常保留详细前缀，但为了与 hide_prefix 选项保持一致性，也应用它。
    LOGGER.addHandler(stream_handler)

    # 初始日志：我们通常希望这条日志是详细的，所以单独处理，或者接受它被隐藏
    # 为了简化，我们让它遵循 hide_prefix 的设置：
    LOGGER.info(f"Logger initialized. Writing logs to: {log_file_path}", stacklevel=2) 
    _LOGGING_SETUP_COMPLETE = True


# --- 以下是兼容多参数拼接和 stacklevel=2 的日志包装函数，与上一个回答相同 ---

def _log_message(level_func, *args, **kwargs):
    if not args:
        return
    msg = " ".join(map(str, args))
    kwargs['stacklevel'] = kwargs.get('stacklevel', 2) 
    level_func(msg, **kwargs)

def debug(*args, **kwargs):
    _log_message(LOGGER.debug, *args, **kwargs)

def info(*args, **kwargs):
    _log_message(LOGGER.info, *args, **kwargs)

def warning(*args, **kwargs):
    _log_message(LOGGER.warning, *args, **kwargs)

def error(*args, **kwargs):
    _log_message(LOGGER.error, *args, **kwargs)

def critical(*args, **kwargs):
    _log_message(LOGGER.critical, *args, **kwargs)