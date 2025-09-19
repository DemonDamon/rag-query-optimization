import asyncio
import aiohttp
import json
import uuid
import time
import os
import base64
import yaml
from datetime import datetime, timedelta
import logging
import sys
from PIL import Image
import io
import argparse  # 添加命令行参数解析


"""
# 快速测试
python test_vl_service_pressure.py --quick

# 指定测试图片
python test_vl_service_pressure.py --image test.png

# 指定测试持续时间
python test_vl_service_pressure.py --minutes 10
"""


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pressure_test.log', encoding='utf-8'),  # 添加utf-8编码
        logging.StreamHandler(sys.stdout)
    ]
)

# 远端服务URL
QWEN25_7B_URL = "http://10.19.16.193:8883/yun/ai/vlm/qwen25/7b/chat"
QWEN2_72B_URL = "http://10.19.16.193:8883/yun/ai/vlm/chat"  # qwen2-vl-72b-instruct-awq

# 测试配置
CONCURRENT_REQUESTS = 2  # 并发请求数
TEST_DURATION_HOURS = 0.2  # 测试持续时间（小时）
REQUEST_TIMEOUT = 180  # 请求超时时间（秒）
MAX_CONSECUTIVE_FAILURES = 5  # 最大连续失败次数
FAILURE_SLEEP_TIME = 60  # 连续失败后的休眠时间（秒）
MAX_EDGE_SIZE = 3500  # 图片最大边长
ENABLE_RESIZE = True  # 是否启用图片缩放
# PROMPT_VERSION = "v4"  # 提示词版本，对应instructions_vl_cls.yaml中的键名
PROMPT_VERSION = "v3"  # 提示词版本，对应instructions.yaml中的键名

# 指定测试图片
TEST_IMAGE = "test.png"

# 从yaml文件加载提示词
def load_prompt_from_yaml(version=PROMPT_VERSION):
    """从YAML文件加载指定版本的提示词"""
    # yaml_path = os.path.join(os.path.dirname(__file__), "instructions_vl_cls.yaml")
    yaml_path = os.path.join(os.path.dirname(__file__), "instructions.yaml")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            instructions = yaml.safe_load(file)
            return instructions.get(version, "")
    except Exception as e:
        logging.error(f"加载提示词文件错误: {str(e)}")
        return ""

# 加载提示词
BASE_PROMPT = load_prompt_from_yaml()
logging.info(f"已加载提示词版本: {PROMPT_VERSION}")

# 如果加载失败，使用默认提示词
if not BASE_PROMPT:
    logging.warning("未能成功加载提示词，使用默认提示词")
    BASE_PROMPT = """请详细分析这页PPT的内容和逻辑结构：
1. 页面整体布局和结构
2. 主要文本内容及其层级关系
3. 图形/图表的类型和作用
4. 元素之间的逻辑关系（如箭头指向、流程图连接等）
5. 关键信息点的总结

请以Markdown格式输出，结构化呈现分析结果。
"""

# 统计数据
class Stats:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.errors = {
            "timeout": 0,  # 504超时
            "oom": 0,     # 内存溢出
            "other": 0    # 其他错误
        }
        self.consecutive_failures = 0
        self.start_time = None
        self.end_time = None

    def log_success(self):
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0

    def log_failure(self, error_type: str):
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.errors[error_type] += 1

    def get_success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def print_summary(self):
        duration = self.end_time - self.start_time if self.end_time and self.start_time else timedelta()
        logging.info("\n=== 压力测试统计摘要 ===")
        logging.info(f"测试开始时间: {self.start_time}")
        logging.info(f"测试结束时间: {self.end_time}")
        logging.info(f"测试持续时间: {duration}")
        logging.info(f"总请求数: {self.total_requests}")
        logging.info(f"成功请求数: {self.successful_requests}")
        logging.info(f"失败请求数: {self.failed_requests}")
        logging.info(f"成功率: {self.get_success_rate():.2f}%")
        logging.info("\n错误分布:")
        logging.info(f"- 超时 (504): {self.errors['timeout']}")
        logging.info(f"- 内存溢出 (OOM): {self.errors['oom']}")
        logging.info(f"- 其他错误: {self.errors['other']}")

def resize_image(image_path: str) -> Image:
    """等比例缩放图片"""
    image = Image.open(image_path)
    width, height = image.size
    if max(width, height) > MAX_EDGE_SIZE:
        scale = MAX_EDGE_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def encode_image(image_path: str) -> str:
    """直接将图片转为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_image(image_path: str) -> str:
    """处理图片并返回base64编码"""
    try:
        if ENABLE_RESIZE:
            # 需要缩放时，先缩放再转base64
            image = resize_image(image_path)
            buffered = io.BytesIO()
            image.save(buffered, format='PNG')
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            # 不需要缩放时，直接转base64
            image_base64 = encode_image(image_path)
        
        # 确保base64编码的安全性和完整性
        # 检查是否是URL安全的base64编码
        if not set(image_base64).issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')):
            image_base64 = image_base64.replace('-', '+').replace('_', '/')

        # 添加缺失的padding
        padding = 4 - (len(image_base64) % 4) if len(image_base64) % 4 else 0
        image_base64 = image_base64 + ('=' * padding)
        
        # 返回不带前缀的base64字符串，前缀会在make_request中添加
        return image_base64
    except Exception as e:
        logging.error(f"处理图片失败: {e}")
        raise

# 预处理测试图片
def prepare_test_image():
    """预处理测试图片并返回base64编码"""
    image_path = TEST_IMAGE
    if not os.path.exists(image_path):
        logging.error(f"测试图片不存在: {image_path}")
        sys.exit(1)
        
    logging.info(f"正在处理测试图片: {image_path}")
    return process_image(image_path)

async def make_request(session: aiohttp.ClientSession, image_base64: str, stats: Stats) -> bool:
    """发送单个请求并处理响应"""
    try:
        payload = {
            "requestId": f"test-{uuid.uuid4().hex[:20]}",
            "model": "qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "imageBase64",
                            "imageBase64": f"data:image;base64,{image_base64}"
                        },
                        {
                            "type": "text",
                            "text": BASE_PROMPT
                        }
                    ]
                }
            ],
            "stream": False,  # 压力测试使用非流式响应
            "maxTokens": 2048,
            "temperature": 0.1,
            "topP": 0.9,
            "frequencyPenalty": 0
        }

        async with session.post(
            QWEN2_72B_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        ) as response:
            if response.status == 504:
                logging.warning(f"请求超时 (504)")
                stats.log_failure("timeout")
                return False

            response_json = await response.json()
            
            # 处理响应内容使中文显示正常
            # 将response_json转为字符串时确保中文不被转义为Unicode序列
            response_text = json.dumps(response_json, ensure_ascii=False)
            
            # 检查响应中是否包含OOM相关信息
            if any(error_text in response_text.lower() for error_text in ["oom", "out of memory", "memory error"]):
                logging.warning(f"检测到内存溢出 (OOM)")
                stats.log_failure("oom")
                return False

            if response.status != 200:
                logging.warning(f"请求失败，状态码: {response.status}")
                stats.log_failure("other")
                return False

            # 检查响应中是否有内容
            content = ""
            if "data" in response_json and "choices" in response_json["data"] and response_json["data"]["choices"]:
                choice = response_json["data"]["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
            
            logging.info(f"请求成功处理 | 响应状态：{response.status} | 响应内容: {content}")
            
            stats.log_success()
            return True

    except asyncio.TimeoutError:
        logging.warning("请求超时")
        stats.log_failure("timeout")
        return False
    except Exception as e:
        logging.error(f"请求异常: {str(e)}")
        stats.log_failure("other")
        return False

async def pressure_test(duration_minutes=None):
    """执行压力测试"""
    stats = Stats()
    stats.start_time = datetime.now()
    
    # 如果提供了分钟数，则用分钟计算，否则使用默认小时数
    if duration_minutes is not None:
        end_time = stats.start_time + timedelta(minutes=duration_minutes)
        logging.info(f"设置测试持续时间为 {duration_minutes} 分钟")
    else:
        end_time = stats.start_time + timedelta(hours=TEST_DURATION_HOURS)

    # 预处理测试图片，只处理一次
    image_base64 = prepare_test_image()

    async with aiohttp.ClientSession() as session:
        while datetime.now() < end_time:
            # 检查连续失败次数
            if stats.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.warning(f"检测到 {MAX_CONSECUTIVE_FAILURES} 次连续失败，休眠 {FAILURE_SLEEP_TIME} 秒")
                await asyncio.sleep(FAILURE_SLEEP_TIME)
                
                # 再次尝试一个请求
                if not await make_request(session, image_base64, stats):
                    logging.error("休眠后仍然失败，终止测试")
                    break
                continue

            # 创建并发请求
            tasks = [make_request(session, image_base64, stats) for _ in range(CONCURRENT_REQUESTS)]
            await asyncio.gather(*tasks)

            # 简单的请求间隔，避免过于密集
            await asyncio.sleep(0.1)

    stats.end_time = datetime.now()
    stats.print_summary()

    # 返回测试是否因为连续失败而终止
    return stats.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VLM服务压力测试工具")
    parser.add_argument("--quick", action="store_true", help="快速测试模式，只运行5分钟")
    parser.add_argument("--minutes", type=int, help="指定测试持续时间（分钟）")
    parser.add_argument("--image", type=str, help="指定测试图片路径")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 处理命令行参数
    test_duration_minutes = None
    if args.quick:
        test_duration_minutes = 5  # 快速测试模式，5分钟
    elif args.minutes:
        test_duration_minutes = args.minutes
    
    if args.image:
        global TEST_IMAGE
        TEST_IMAGE = args.image
    
    # 打印配置信息
    logging.info("开始压力测试...")
    logging.info(f"配置信息:")
    logging.info(f"- 并发请求数: {CONCURRENT_REQUESTS}")
    if test_duration_minutes:
        logging.info(f"- 测试持续时间: {test_duration_minutes}分钟")
    else:
        logging.info(f"- 测试持续时间: {TEST_DURATION_HOURS}小时")
    logging.info(f"- 请求超时时间: {REQUEST_TIMEOUT}秒")
    logging.info(f"- 测试图片: {TEST_IMAGE}")
    logging.info(f"- 提示词版本: {PROMPT_VERSION}")
    
    # 执行测试
    terminated_by_failures = asyncio.run(pressure_test(test_duration_minutes))
    
    if terminated_by_failures:
        logging.error("测试因连续失败而终止")
    else:
        logging.info("测试正常完成")

if __name__ == "__main__":
    main()
