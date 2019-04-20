import os.path
def output_file(output_dir, phase, checkpoint):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, phase + '_' + checkpoint)

