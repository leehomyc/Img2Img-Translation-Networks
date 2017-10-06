"""Creating a file of image names given a directory."""
import os

import click
import numpy as np
import pandas as pd


@click.command()
@click.option('--input_path',
              type=click.Path(),
              default=None,
              help='a string as the input directory.')
@click.option('--extension',
              type=click.STRING,
              default='.jpg',
              help='a string as the extension of the file.')
@click.option('--output_file',
              type=click.Path(),
              default=None,
              help='path of the output file.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=True,
              help='whether to shuffle the dataset.')
def create_list(input_path, extension, output_file, do_shuffle):
    """ Create a file that contains the list of file names."""
    random = np.random.RandomState(seed=42)

    filelist = []
    for fullname in os.listdir(input_path):
        if fullname.endswith(extension):
            fullpath = os.path.join(input_path, fullname)
            filelist.append(fullpath)
    if do_shuffle is True:
        random.shuffle(filelist)
    df = pd.DataFrame(filelist)
    df.to_csv(output_file, header=False, index=False)

if __name__ == '__main__':
    create_list()
