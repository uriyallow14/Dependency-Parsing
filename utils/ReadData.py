def get_data(version):
    basic_ver = None
    if version == 'basic':
        basic_ver = True
    elif version == 'advanced':
        basic_ver = False
    else:
        KeyError("argument can be either 'basic' or 'advanced'")

    data_dir = 'data\\'
    path_train = data_dir + "train.labeled"
    print("path_train -", path_train)
    path_test = data_dir + "test.labeled"
    print("path_test -", path_test)

    paths_list = [path_train]

    return paths_list, basic_ver, data_dir
