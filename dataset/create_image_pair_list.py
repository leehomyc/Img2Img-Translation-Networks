"""Create the csv file of image pairs to train CycleGAN."""
import logging

import click
import pandas as pd

from cli import configure_script

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@click.command()
@click.option('--first_dataset',
              type=click.Path(),
              default=None,
              help='the first dataset csv file.')
@click.option('--second_dataset',
              type=click.Path(),
              default=None,
              help='the second dataset csv file.')
@click.option('--output_file',
              type=click.Path(),
              default=None,
              help='the output csv file.')
@click.option('--main_dataset',
              type=click.INT,
              default=1,
              help='the main dataset to determine the number of rows.')
def create_dataset(first_dataset, second_dataset, output_file, main_dataset):
    """Create a csv file to train cyclegan.

    Args:
        first_dataset: a string as the path to the first dataset.
        second_dataset: a string as the path to the second dataset.
        output_file: a string as the path to the output file.
        main_dataset: an integer as the main dataset.
    """
    d1 = pd.read_csv(first_dataset, header=None)
    d2 = pd.read_csv(second_dataset, header=None)

    if main_dataset == 1:
        num_rows = len(d1[0])
    else:
        num_rows = len(d2[0])
    all_data_tuples = []
    for i in range(num_rows):
        all_data_tuples.append((
            d1[0][i % len(d1[0])],
            d2[0][i % len(d2[0])],
        ))

    df = pd.DataFrame(all_data_tuples)
    df.to_csv(output_file, header=False, index=False)

    logger.info('Total number of image pairs :{}'.format(num_rows))


if __name__ == '__main__':
    configure_script()
    create_dataset()
