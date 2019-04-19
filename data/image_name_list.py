def get_image_name_list(dataset_file):
    # get synme image name list
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
        imagenames = [x.strip().split()[0].split('/')[1].split('.')[0] for x in lines]
        return imagenames
