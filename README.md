# Occluded Facial Image Retrieval and Reconstruction

This project leverages machine learning techniques to enhance facial image retrieval and reconstruction, even when input images are partially occluded. Using a combination of GANs, image retrieval models, and metric evaluation, this system can retrieve and restore facial features from obscured images effectively.
![Alt Text](https://github.com/ethanbrain/FaceInpaint/blob/main/Demo.png)

## Features
- **GAN-based Image Completion**: Utilizes a Generative Adversarial Network to reconstruct occluded facial images.
- **Image Retrieval**: Efficiently retrieves images similar to the target, assisting in facial recognition tasks.
- **Custom Model Training**: Trains a model from scratch for specific retrieval tasks with tailored metrics.(Be Soon)
- **Inpainting**: Fills missing or occluded parts of images for improved analysis.

## Installation
1. Clone the repository.
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download `model.pth` from the provided link and place it in the project folder.

## Usage
To run the main application:
```bash
python App_Retrieval_Occluded_Facial_Image.py
```

## Example
Below is a comparison of input and output images generated by the project:
- **Input**: Partially occluded image
- **Output**: Fully reconstructed image

## Future Improvements
- Enhance GAN model performance for more realistic reconstructions.
- Integrate more efficient retrieval algorithms.

## Note
The current results were obtained using a model trained with only 150 epochs. While these initial findings demonstrate the model's potential, they are limited by the computational resources available. With more powerful hardware and advanced computational facilities, we could increase the number of training epochs, which would allow the model to learn more complex patterns and improve its performance. Enhanced hardware would also enable us to explore larger batch sizes, deeper network architectures, and more advanced hyperparameter tuning, all of which could contribute to achieving significantly better results.
