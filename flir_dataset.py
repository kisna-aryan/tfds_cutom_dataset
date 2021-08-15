"""flir_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import collections
import json
import os

from absl import logging

# TODO(flir_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(flir_dataset): BibTeX citation
_CITATION = """
"""
class AnnotationType(object):
  """Enum of the annotation format types.

  Splits are annotated with different formats.
  """
  BBOXES = 'bboxes'
  PANOPTIC = 'panoptic'
  NONE = 'none'

class FlirDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for flir_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(flir_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence({
                  'area': tf.int64,
                  'bbox': tfds.features.BBoxFeature(),
                  'id': tf.int64,
                  'is_crowd': tf.bool,
                  'label': tfds.features.ClassLabel(num_classes=81),
    }),
}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # features=tfds.features.FeaturesDict(features),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )



  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(flir_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    data_dir_train = "K:\\DL_git\\FLIRrgb_keras\\train"    
    data_dir_val = "K:\\DL_git\\FLIRrgb_keras\\val"

    # TODO(flir_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(data_dir_train),
        'val': self._generate_examples(data_dir_val),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(flir_dataset): Yields (key, example) tuples from the dataset
    annotation_type = AnnotationType.BBOXES
    instance_path = os.path.join(
        path,
        'thermal_annotations.json',
    )

    coco_annotation = ANNOTATION_CLS[annotation_type](instance_path)
    # Each category is a dict:
    # {
    #    'id': 51,  # From 1-91, some entry missing
    #    'name': 'bowl',
    #    'supercategory': 'kitchen',
    # }
    categories = coco_annotation.categories
    # Each image is a dict:
    # {
    #     'id': 262145,
    #     'file_name': 'COCO_train2017_000000262145.jpg'
    #     'flickr_url': 'http://farm8.staticflickr.com/7187/xyz.jpg',
    #     'coco_url': 'http://images.cocodataset.org/train2017/xyz.jpg',
    #     'license': 2,
    #     'date_captured': '2013-11-20 02:07:55',
    #     'height': 427,
    #     'width': 640,
    # }
    images = coco_annotation.images

    # TODO(b/121375022): ClassLabel names should also contains 'id' and
    # and 'supercategory' (in addition to 'name')
    # Warning: As Coco only use 80 out of the 91 labels, the c['id'] and
    # dataset names ids won't match.

    objects_key = 'objects'
    self.info.features[objects_key]['label'].names = [
        c['name'] for c in categories
    ]
    # TODO(b/121375022): Conversion should be done by ClassLabel
    categories_id2name = {c['id']: c['name'] for c in categories}

    # Iterate over all images
    annotation_skipped = 0
    for image_info in sorted(images, key=lambda x: x['id']):
      if annotation_type == AnnotationType.BBOXES:
        # Each instance annotation is a dict:
        # {
        #     'iscrowd': 0,
        #     'bbox': [116.95, 305.86, 285.3, 266.03],
        #     'image_id': 480023,
        #     'segmentation': [[312.29, 562.89, 402.25, ...]],
        #     'category_id': 58,
        #     'area': 54652.9556,
        #     'id': 86,
        # }
        instances = coco_annotation.get_annotations(img_id=image_info['id'])
      else:
        instances = []  # No annotations

      if not instances:
        annotation_skipped += 1

      def build_bbox(x, y, width, height):
        # pylint: disable=cell-var-from-loop
        # build_bbox is only used within the loop so it is ok to use image_info
        return tfds.features.BBox(
            ymin=y / image_info['height'],
            xmin=x / image_info['width'],
            ymax=(y + height) / image_info['height'],
            xmax=(x + width) / image_info['width'],
        )
        # pylint: enable=cell-var-from-loop

      example = {
          'image': os.path.join(path, image_info['file_name']),
          'image/filename': image_info['file_name'],
          'image/id': image_info['id'],
          objects_key: [{   # pylint: disable=g-complex-comprehension
              'id': instance['id'],
              'area': instance['area'],
              'bbox': build_bbox(*instance['bbox']),
              'label': categories_id2name[instance['category_id']],
              'is_crowd': bool(instance['iscrowd']),
          } for instance in instances]
      }
      yield image_info['file_name'], example

    logging.info(
        '%d/%d images do not contains any annotations',
        annotation_skipped,
        len(images),
    )


class CocoAnnotation(object):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    with tf.io.gfile.GFile(annotation_path) as f:
      data = json.load(f)
    self._data = data

  @property
  def categories(self):
    """Return the category dicts, as sorted in the file."""
    return self._data['categories']

  @property
  def images(self):
    """Return the image dicts, as sorted in the file."""
    return self._data['images']

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    raise NotImplementedError  # AnotationType.NONE don't have annotations

class CocoAnnotationBBoxes(CocoAnnotation):
  """Coco annotation helper class."""

  def __init__(self, annotation_path):
    super(CocoAnnotationBBoxes, self).__init__(annotation_path)

    img_id2annotations = collections.defaultdict(list)
    for a in self._data['annotations']:
      img_id2annotations[a['image_id']].append(a)
    self._img_id2annotations = {
        k: list(sorted(v, key=lambda a: a['id']))
        for k, v in img_id2annotations.items()
    }

  def get_annotations(self, img_id):
    """Return all annotations associated with the image id string."""
    # Some images don't have any annotations. Return empty list instead.
    return self._img_id2annotations.get(img_id, [])


ANNOTATION_CLS = {
    AnnotationType.NONE: CocoAnnotation,
    AnnotationType.BBOXES: CocoAnnotationBBoxes,     
}