# Face Tagger

Face Tagger is a Python library designed to classify photos containing specific individuals from a collection of images.

<br>

## Features

- **Face Recognition**: Identify and differentiate faces in multiple images.
- **Photo Classification**: Group photos based on the identified individual.
- **GPU Acceleration**: Optimized performance using GPU, if available.
- **Easy Integration**: Simple APIs for seamless integration into various applications.

<br>

## Installation

To install Face Tagger, you can use pip:

```bash
pip install face-tagger
```

<br>

## Dependencies

Face Tagger relies on several key libraries to function effectively. Understanding these dependencies can provide insights into the library's internal workings and can be beneficial for troubleshooting. Here are the primary dependencies:

- **OpenCV (cv2)**: Used for image processing tasks like reading and preprocessing images.
- **Torch (torch)**: The primary deep learning framework employed for face embedding.
- **MTCNN**: Essential for the detection of faces within images.

Make sure to have these dependencies properly installed or ensure they are present in your environment when working with Face Tagger. For a comprehensive list and exact versions, please refer to the `requirements.txt` file.

<br>

## How it Works

Face Tagger follows a systematic process to classify images:

1. **Image Loading**
    - Uses standard Python libraries, typically OpenCV (`cv2`), to read and preprocess images.
    
2. **Face Detection**
    - Utilizes the MTCNN (Multi-task Cascaded Convolutional Networks) library for detecting faces in images.

3. **Embedding Computation**
    - Employs the FaceNet model, often via the `torch` library, to compute a vector (embedding) for each
      detected face.

4. **Face Classification**
    - Uses a HDBSCAN clustering algorithm to group similar face embeddings together.

5. **Result Compilation**
    - Once the faces are classified, the result is compiled and presented in a structured format that informs which
      images contain which individual(s).

<br>

## Usage

To utilize the Face Tagger library, follow these steps:

1. **Load image with functions defined in utils module**:
   The generator function allows you to loop through all images in a specified directory and create `ImageObject`
   instances for each one.

```python
from face_tagger.utils import *

image_objects = load_images_from_directory("path_to_images_directory")
```

Replace `path_to_images_directory` with the path to your image directory.

(Note: You don't necessarily need to use a generator for image_objects. You can also create ImageObject list from loaded
images in an array and pass them.)

Here is the representation of the `ImageObject` class:

> #### Attributes:
> - **image_id**: A unique identifier for the image.
> - **image_data**: Actual image data, loaded using OpenCV.
>
> #### Example:
> ```python
> img_obj = ImageObject(image_id="unique_id_123", image_data=loaded_image_data)
> ```

2. **Initialize the Face Tagger**:
   Here, you can set various parameters like `use_gpu`, `image_resize_factor`, `min_faces_to_identify_human`,
   and `min_similarity_face_count`.

- **`min_faces_to_identify_human`**: Minimum number of faces required to be classified as a human.
- **`min_similarity_face_count`**: Minimum number of similar faces required to be included in a group.

```python
from face_tagger import FaceTagger

face_tagger = FaceTagger(
    use_gpu=False,
    image_resize_factor=1.0,
    min_faces_to_identify_human=4,
    min_similarity_face_count=3
)
```

3. **Classify images by person**:
   Now, use the `classify_images_by_person` method of the `face_tagger` instance and pass the `image_objects` from the
   generator function to classify the images.

```python
result = face_tagger.classify_images_by_person(image_objects=image_objects)
print(result)
```

---

With these steps, you can effectively utilize the Face Tagger library to classify images based on the individuals they
contain.

<br>

## Contributing

Pull requests are welcome. For major changes, please open an [issue]('https://github.com/Face-Tagger/facetagger-lib/issues/new') first to discuss what you would like to change.

<br>

## License

[MIT](https://github.com/Face-Tagger/facetagger-lib/blob/main/LICENSE)
