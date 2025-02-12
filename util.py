import logging
from colorlog import ColoredFormatter
import os

# 配置彩色日志输出处理器
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 配置彩色日志格式
formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",  # 日志格式
#    datefmt='%H:%M:%S.%f',  # 时间格式（已注释）
    datefmt=None,  # 使用默认时间格式
    reset=True,  # 重置颜色
    log_colors={
        'DEBUG':    'cyan',  # DEBUG级别颜色
        'INFO':     'white,bold',  # INFO级别颜色
        'INFOV':    'cyan,bold',  # INFOV级别颜色
        'WARNING':  'yellow',  # WARNING级别颜色
        'ERROR':    'red,bold',  # ERROR级别颜色
        'CRITICAL': 'red,bg_white',  # CRITICAL级别颜色
    },
    secondary_log_colors={},  # 二级颜色配置
    style='%'  # 使用%格式化
)
ch.setFormatter(formatter)  # 将格式应用到处理器

# 配置日志记录器
log = logging.getLogger('rn')  # 创建名为'rn'的日志记录器
log.setLevel(logging.DEBUG)  # 设置日志级别
log.handlers = []       # 清除已有处理器，避免重复
log.propagate = False   # 防止在ipython中出现重复日志
log.addHandler(ch)  # 添加配置好的处理器

def check_path(path):
    """检查路径是否存在，如果不存在则创建
    Args:
        path (str): 要检查的路径
    """
    if not os.path.exists(path):
        os.mkdir(path)
