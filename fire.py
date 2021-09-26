"""fire dataset."""

import tensorflow_datasets as tfds
import os
import glob

# TODO(fire): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(fire): BibTeX citation
_CITATION = """
"""


class Fire(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for fire dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(fire): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['Fire', 'No_Fire']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(fire): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path_train = f'/media/kisna/nano_ti_data/DL_git/keras_classification/fire_classification/fire_class_dataset/Training'
    path_val = f'/media/kisna/nano_ti_data/DL_git/keras_classification/fire_classification/fire_class_dataset/Test'


    # TODO(fire): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path_train),
        'val': self._generate_examples(path_val),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(fire): Yields (key, example) tuples from the dataset
    subfolders = [ f.name for f in os.scandir(path) if f.is_dir() ]
    for folder in subfolders:
      path_new = os.path.join(path, folder)
      for f in glob.glob(path_new + '/' + '*.jpg'):
        yield 'key', {
            'image': f,
            'label': folder,
        }
