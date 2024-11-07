import os

dir_path = "./graphs/cond"

if os.path.isdir(dir_path):
    print("目录存在")
else:
    print("目录不存在")