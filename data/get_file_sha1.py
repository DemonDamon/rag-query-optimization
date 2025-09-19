import argparse
import hashlib

def sha1(fname):
    hash_sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='计算文件的SHA1值')
    parser.add_argument('--file', required=True, type=str, help='要计算SHA1值的文件路径')
    args = parser.parse_args()
    
    sha1_hash = sha1(args.file)
    print(f"文件 {args.file} 的SHA1值为: {sha1_hash}")

if __name__ == "__main__":
    main()