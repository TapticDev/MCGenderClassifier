# MCGenderClassifier

This Python script uses TensorFlow and Custom Tkinter to predict the gender of a Minecraft Skin.

## Usage

1. Run the script.
2. Click the "Upload Skin Image" button to load and predict the gender.
3. The predicted gender and confidence level will be displayed.

## Dependencies

- `PIL` (Pillow)
- `customtkinter`
- `tensorflow==2.9.3`
- `numpy`


## Installation

1. Install the required dependencies using `pip install pillow customtkinter tensorflow==2.9.3 numpy`.
2. Run the script with `python mcgc.py`.

## Notes

- The trained model (`classifier.h5`) should be present in the same directory as the script.

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
