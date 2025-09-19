# Date    : 2024/7/22 13:39
# File    : utils.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import yaml
import json
import socket
import argparse


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


def read_yaml(file_path):
    """
    读取并解析一个YAML文件。

    :param file_path: YAML文件的绝对路径。
    :return: 解析后的数据结构。
    """
    # # 验证文件路径参数
    # if not os.path.isabs(file_path) or '..' in file_path:
    #     # 以防止路径遍历攻击
    #     raise ValueError("文件路径不合法，禁止使用相对路径或 '..' ")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"无法找到文件: {file_path}")
    except PermissionError:
        raise PermissionError(f"没有权限读取文件: {file_path}")
    except yaml.YAMLError as e:
        # 可以进一步细化YAML错误处理
        raise ValueError(f"YAML文件格式错误: {file_path}") from e

    return data


class Env(object):
    """Helper class to get Env variables"""

    @classmethod
    def get_value(cls, config_name, default="", value_type=str):
        value = os.getenv(config_name, default)

        if value_type is str:
            return str(value)
        elif value_type is int:
            return int(value)
        elif value_type is bool:
            return value.lower() in ['true', '1', 't', 'y', 'yes']
        else:
            return value


class Computer(object):
    import socket

    IP = ''
    NAME = ''
    # noinspection PyBroadException
    try:
        sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sc.connect(('8.8.8.8', 80))
        IP = sc.getsockname()[0]
        NAME = socket.gethostname()
        sc.close()
    except Exception as e:
        pass


def get_host_ip():
    ip = ''
    host_name = ''
    try:
        sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sc.connect(('8.8.8.8', 80))
        ip = sc.getsockname()[0]
        host_name = socket.gethostname()
        sc.close()
    except Exception:
        pass
    return ip, host_name


def generate_directory_structure(directory_path: str, output_file: str, indent: str = "    "):
    """
    生成目录结构并保存到文件
    
    Args:
        directory_path: 要扫描的目录路径
        output_file: 输出文件路径
        indent: 缩进字符串，默认4个空格
    
    Example:
        generate_directory_structure("./project", "structure.txt")
        
        # 生成的结构示例：
        # project/
        #     ├── src/
        #     │   ├── models.py
        #     │   ├── utils.py
        #     │   └── api.py
        #     ├── tests/
        #     │   └── test_api.py
        #     └── README.md
    """
    def get_tree(dir_path: str, prefix: str = "") -> list:
        """递归获取目录结构"""
        entries = []
        # 获取目录内容并排序
        items = os.listdir(dir_path)
        items.sort()
        
        # 分离文件和目录
        files = [x for x in items if os.path.isfile(os.path.join(dir_path, x))]
        dirs = [x for x in items if os.path.isdir(os.path.join(dir_path, x))]
        
        # 处理目录
        for i, dir_name in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1 and not files)
            current = os.path.join(dir_path, dir_name)
            
            # 添加目录名
            entries.append(f"{prefix}{'└──' if is_last_dir else '├──'} {dir_name}/")
            
            # 递归处理子目录
            extension = "    " if is_last_dir else "│   "
            entries.extend(get_tree(current, prefix + extension))
        
        # 处理文件
        for i, file_name in enumerate(files):
            is_last = (i == len(files) - 1)
            entries.append(f"{prefix}{'└──' if is_last else '├──'} {file_name}")
        
        return entries

    try:
        # 确保目录存在
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 获取目录名
        base_name = os.path.basename(directory_path.rstrip('/\\'))
        
        # 生成目录结构
        structure = [f"{base_name}/"]
        structure.extend(get_tree(directory_path, indent))
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(structure))
            
        print(f"目录结构已保存到: {output_file}")
        
    except Exception as e:
        print(f"生成目录结构时出错: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='生成目录结构树')
    parser.add_argument('directory', nargs='?', default='.', 
                      help='要扫描的目录路径 (默认: 当前目录)')
    parser.add_argument('--output', '-o', default='directory_structure.txt',
                      help='输出文件路径 (默认: directory_structure.txt)')
    parser.add_argument('--indent', '-i', default='    ',
                      help='缩进字符串 (默认: 4个空格)')
    
    args = parser.parse_args()
    
    # 生成目录结构
    generate_directory_structure(args.directory, args.output, args.indent)

if __name__ == "__main__":
    main()
