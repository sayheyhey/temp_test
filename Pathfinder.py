import os




"""
功能：生成文件夹
输入：想要生成的路径的文件夹名
"""
def my_mkdir(path_str):
    path_list = []
    while path_str != '/':
        path_str, tmp_path = os.path.split(path_str)
        path_list.insert(0, tmp_path)

    abs_path = os.path.abspath('.')
    for path in path_list:
        abs_path = os.path.join(abs_path, path)
        if not os.path.exists(abs_path):
            os.mkdir(abs_path)



