# MCGenderClassifier

This Python script uses TensorFlow and Tkinter to predict the gender of a Minecraft player based on their in-game skin. Additionally, it provides a feature to report false positives.

## Usage

1. Run the script.
2. Enter a player's Minecraft name.
3. Click the "Load Skin" button to load and predict the gender.
4. The predicted gender and confidence level will be displayed.
5. A cube render (body render) of the player will be shown on the right.
6. Optionally, click "Report as False Positive" to save the skin as a false positive.

## Dependencies

- `tkinter`
- `PIL` (Pillow)
- `requests`
- `tensorflow`
- `numpy`

## Installation

1. Install the required dependencies using `pip install tkinter pillow requests tensorflow numpy`.
2. Run the script with `python script_name.py`.

## Notes

- The trained model (`minecraft_gender_classifier.h5`) should be present in the same directory as the script.
- Ensure an internet connection for fetching player information and skins.

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
